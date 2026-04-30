import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import open_clip
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import ast

# Ensure the models are loaded
from multimodal_experiment import MultimodalMHISTDataset, histology_transform, MultimodalQuiltClassifier

MODEL_NAME = "hf-hub:wisdomik/QuiltNet-B-32"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGES_DIR = "images"

def get_attention_map(model, image_tensor, device, suppress_sinks=False):
    """
    Extracts the self-attention map from the last layer of the Vision Transformer.
    For ViT-B/32, this results in a 7x7 grid of attention scores.
    """
    attentions = []
    
    def hook(module, input, output):
        # In this specific QuiltNet model, batch_first=True is used for the ViT!
        # x is (Batch, Tokens, Dim)
        x = input[0] 
        q, k, v = F.linear(x, module.in_proj_weight, module.in_proj_bias).chunk(3, dim=-1)

        batch, tokens, dim = x.shape
        num_heads = module.num_heads
        head_dim = dim // num_heads

        # Reshape to (batch, num_heads, tokens, head_dim)
        q = q.view(batch, tokens, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch, tokens, num_heads, head_dim).transpose(1, 2)

        # Calculate attention: softmax(QK^T / sqrt(d))
        attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attentions.append(attn.detach().cpu())

    # Register hook on the last attention block
    handle = model.base_model.visual.transformer.resblocks[-1].attn.register_forward_hook(hook)

    with torch.no_grad():
        model.base_model.encode_image(image_tensor.to(device))

    handle.remove()

    # Average across all heads
    attn = attentions[0].mean(dim=1) # Shape: (batch, seq_len, seq_len)

    # Extract CLS token attention (index 0) to all spatial patches (indices 1 to 49)
    cls_attn = attn[:, 0, 1:]
    
    if suppress_sinks:
        # Test-Time Intervention: Attention Sink Suppression
        # Mimics the visual result of "Vision Transformers Don't Need Trained Registers" 
        # by statistically identifying and suppressing extreme outliers (the sinks).
        mean_val = cls_attn.mean(dim=1, keepdim=True)
        std_val = cls_attn.std(dim=1, keepdim=True)
        
        # Identify statistical outliers and cap them to a smoothed threshold
        threshold = mean_val + 2.5 * std_val
        cls_attn = torch.where(cls_attn > threshold, threshold, cls_attn)

    # Reshape to 7x7 grids (for ViT-B/32)
    grid_size = int(np.sqrt(cls_attn.shape[-1]))
    return cls_attn.reshape(-1, grid_size, grid_size)

def get_text_attention_map(model, text_tokens, tokenizer):
    attentions = []
    def hook(module, input, output):
        # OpenCLIP text transformer uses (L, B, E) layout
        x = input[0] 
        q, k, v = F.linear(x, module.in_proj_weight, module.in_proj_bias).chunk(3, dim=-1)
        # QuiltNet Text Transformer ALSO uses batch_first=True
        batch, seq_len, dim = x.shape
        num_heads = module.num_heads
        head_dim = dim // num_heads

        q = q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attentions.append(attn.detach().cpu())

    handle = model.base_model.transformer.resblocks[-1].attn.register_forward_hook(hook)
    with torch.no_grad():
        model.base_model.encode_text(text_tokens)
    handle.remove()

    attn = attentions[0].mean(dim=1) # (batch, seq_len, seq_len)
    eos_idx = text_tokens[0].argmax().item()
    eos_attention = attn[0, eos_idx, 1:eos_idx].numpy()
    
    words = []
    for token in text_tokens[0][1:eos_idx]:
        word = tokenizer.decode([token.item()]).replace('</w>', '')
        words.append(word)
        
    return eos_attention, words

def show_samples_with_attention(df, model, preprocess, tokenizer, num_samples=3, save_path="attention_samples.png", suppress_sinks=False):
    num_rows = num_samples
    plt.figure(figsize=(15, 4 * num_rows)) # Increased width for 3rd column

    correct_samples = []
    incorrect_samples = []
    
    model.eval()
    for idx, row in df.sample(frac=1, random_state=42).iterrows():
        img_path = os.path.join(IMAGES_DIR, row['image'])
        raw_img = Image.open(img_path).convert("RGB")
        img_tensor = preprocess(raw_img).unsqueeze(0).to(device)
        
        prompts = ast.literal_eval(row['top_prompts'])
        text_tokens = tokenizer([prompts[0]]).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor, text_tokens)
            pred_idx = torch.argmax(outputs, dim=1).item()
            pred_label = "HP" if pred_idx == 1 else "SSA"
            
        true_label = row.get('Majority Vote Label', 'Unknown')
        
        if pred_label == true_label:
            if len(correct_samples) < num_samples - 1:
                correct_samples.append((row, raw_img, img_tensor, text_tokens, pred_label))
        else:
            if len(incorrect_samples) < 1:
                incorrect_samples.append((row, raw_img, img_tensor, text_tokens, pred_label))
                
        if len(correct_samples) == num_samples - 1 and len(incorrect_samples) == 1:
            break
            
    selected_samples = correct_samples + incorrect_samples

    for i, (row, raw_img, img_tensor, text_tokens, pred_label) in enumerate(selected_samples):
        # Get attention maps
        attn_map = get_attention_map(model, img_tensor, device, suppress_sinks=suppress_sinks)[0]
        text_attn, words = get_text_attention_map(model, text_tokens, tokenizer)

        prompts = ast.literal_eval(row['top_prompts'])
        top_prompt = prompts[0]
        true_label = row.get('Majority Vote Label', 'Unknown')

        # Subplot 1: Original Image
        plt.subplot(num_rows, 3, 3*i + 1)
        plt.imshow(raw_img)
        title_color = "red" if pred_label != true_label else "black"
        plt.title(f"True: {true_label} | Pred: {pred_label}\nPrompt: {top_prompt}", color=title_color)
        plt.axis('off')

        # Subplot 2: Focus Heatmap Overlay
        plt.subplot(num_rows, 3, 3*i + 2)
        plt.imshow(raw_img)
        heatmap = F.interpolate(attn_map.unsqueeze(0).unsqueeze(0),
                                size=raw_img.size[::-1], mode='bicubic').squeeze()
        heatmap_np = heatmap.numpy()
        heatmap_np = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-8)
        plt.imshow(heatmap_np, cmap='jet', alpha=0.5)
        plt.title("Model Focus Area (ROI)")
        plt.axis('off')
        
        # Filter out punctuation "attention sinks" to see the true semantic weights
        valid_indices = [j for j, w in enumerate(words) if w.strip() not in ['.', ',', ';', ':', '!', '?', '(', ')', '-'] and w.strip() != '']
        filtered_words = [words[j] for j in valid_indices]
        filtered_text_attn = text_attn[valid_indices]
        
        # Subplot 3: Text Attention Bar Chart
        ax3 = plt.subplot(num_rows, 3, 3*i + 3)
        text_attn_norm = (filtered_text_attn - filtered_text_attn.min()) / (filtered_text_attn.max() - filtered_text_attn.min() + 1e-8)
        
        # Use 'plasma' colormap: Dark Purple (Low) -> Orange -> Bright Yellow (High)
        colors = plt.cm.plasma(text_attn_norm)
        ax3.barh(np.arange(len(filtered_words)), filtered_text_attn, color=colors)
        ax3.set_yticks(np.arange(len(filtered_words)))
        ax3.set_yticklabels(filtered_words)
        ax3.invert_yaxis()  # Read top-to-bottom
        ax3.set_title("Text Prompt Attention Weights")
        ax3.set_xlabel("Attention Weight\n(Dark Purple = Low, Bright Yellow = High)")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved {num_samples}-sample attention comparison to {save_path}")

if __name__ == "__main__":
    print("Loading model and utilities...")
    base_model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=True)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    
    # Initialize with the exact optimal dropout rate from the #1 rank experiment
    classifier = MultimodalQuiltClassifier(base_model, dropout_rate=0.5110907127277374).to(device)
    
    # Load the best saved checkpoint
    best_model_path = "results/finetuned_top1/mhist_with_captions_all_prompts/best_study_model.pth"
    classifier.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    print(f"Loaded optimal weights from {best_model_path}")
    
    # Load dataset
    df = pd.read_csv("linus_branch/captions/mhist_with_captions_all_prompts.csv")
    
    # Merge with annotations to get the True Label
    annotations = pd.read_csv("data/annotations.csv")
    df = df.merge(annotations, left_on="image", right_on="Image Name", how="inner")
    
    # Filter strictly for Sessile Serrated Adenoma (SSA) cases
    ssa_df = df[df['Majority Vote Label'] == 'SSA']
    show_samples_with_attention(ssa_df, classifier, preprocess, tokenizer, num_samples=3, save_path="attention_samples_ssa.png")
    show_samples_with_attention(ssa_df, classifier, preprocess, tokenizer, num_samples=3, save_path="attention_samples_ssa_smoothed.png", suppress_sinks=True)
    
    # Filter strictly for Hyperplastic Polyp (HP) cases
    hp_df = df[df['Majority Vote Label'] == 'HP']
    show_samples_with_attention(hp_df, classifier, preprocess, tokenizer, num_samples=3, save_path="attention_samples_hp.png")
    show_samples_with_attention(hp_df, classifier, preprocess, tokenizer, num_samples=3, save_path="attention_samples_hp_smoothed.png", suppress_sinks=True)
