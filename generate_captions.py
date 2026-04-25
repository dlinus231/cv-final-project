import os
import re
import pandas as pd
import torch
from transformers import AutoProcessor, BlipForConditionalGeneration, LlavaForConditionalGeneration
from PIL import Image
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-mode", action="store_true", help="Only run on 15 images to preview.")
    parser.add_argument("--model", type=str, choices=["blip", "llava"], default="llava", help="Choose the VLM to use.")
    # Batch size argument to speed up inference massively
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for generating captions.")
    args = parser.parse_args()

    ANNOTATIONS_PATH = "annotations.csv"
    IMAGES_DIR = "images"
    OUTPUT_PATH = "annotations_captioned.csv"

    print("Loading Dataset...")
    df = pd.read_csv(ANNOTATIONS_PATH)
    if args.test_mode:
        df = df.head(15).copy()
        print("Test mode: Only processing 15 images.")
    else:
        print(f"Processing all {len(df)} images.")

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.model == "blip":
        MODEL_ID = "Salesforce/blip-image-captioning-base"
        print(f"Loading {MODEL_ID} on {device} in float16...")
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        # Using SDPA (Scaled Dot-Product Attention) explicitly utilizes Apple's native hardware ML optimizations
        model = BlipForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, attn_implementation="sdpa"
        )
        model.to(device)
        prompt = "A histopathology slide showing"
        
    elif args.model == "llava":
        MODEL_ID = "llava-hf/llava-interleave-qwen-0.5b-hf"
        print(f"Loading {MODEL_ID} on {device} in float16...")
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        
        # Explicitly enabling hardware-accelerated SDPA Attention
        model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, attn_implementation="sdpa"
        )
        model.to(device)
        raw_prompt = "You are an expert pathologist. Look at this H&E stained histopathology slide. Describe the visual layout, focusing strictly on glandular architecture, nuclear features, and stroma. Do not use the words 'Polyp', 'Adenoma', 'Hyperplastic', or 'SSA'."

    forbidden_words = [r'\bpolyp\b', r'\badenoma\b', r'\bhyperplastic\b', 
                       r'\bsessile\b', r'\bssa\b', r'\bhp\b']

    captions = [""] * len(df)
    batch_size = args.batch_size
    
    print(f"\nGenerating captions using Batch Size: {batch_size}...")
    
    # Process images in memory-efficient batches!
    for i in tqdm(range(0, len(df), batch_size), desc="Processing Batches"):
        batch_df = df.iloc[i : i + batch_size]
        
        valid_images = []
        valid_indices = []
        
        for idx_in_batch, row in batch_df.iterrows():
            img_path = os.path.join(IMAGES_DIR, row['Image Name'])
            try:
                # Store the absolute dataframe index
                idx = i + idx_in_batch if isinstance(idx_in_batch, int) else idx_in_batch
                if isinstance(idx, str) or idx_in_batch > batch_size: 
                    idx = list(df.index).index(idx_in_batch)
                    
                image = Image.open(img_path).convert("RGB")
                valid_images.append(image)
                valid_indices.append(idx)
            except Exception as e:
                print(f"Error opening {img_path}: {e}")
                continue

        if not valid_images:
            continue

        if args.model == "blip":
            texts = [prompt] * len(valid_images)
            inputs = processor(images=valid_images, text=texts, return_tensors="pt", padding=True).to(device, torch.float16)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, 
                    max_new_tokens=60,
                    pad_token_id=processor.tokenizer.pad_token_id
                )
            
            batch_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            
        elif args.model == "llava":
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": raw_prompt},
                    ],
                }
            ]
            
            text_template = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            texts = [text_template] * len(valid_images)
            
            # The processor might not have a pad token for certain models, assign it safely:
            if processor.tokenizer.pad_token is None:
                processor.tokenizer.pad_token = processor.tokenizer.eos_token
                
            inputs = processor(images=valid_images, text=texts, return_tensors="pt", padding=True).to(device, torch.float16)
            
            with torch.no_grad():
                # Setting use_cache=True explicitly speeds up autoregressive generation greatly
                generated_ids = model.generate(
                    **inputs, 
                    max_new_tokens=60, 
                    use_cache=True,
                    pad_token_id=processor.tokenizer.pad_token_id
                )
                
            input_length = inputs["input_ids"].shape[1]
            batch_texts = processor.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)

        for text, real_idx in zip(batch_texts, valid_indices):
            generated_text = text.strip()
            # Guarantee Redaction
            for word_regex in forbidden_words:
                generated_text = re.sub(word_regex, "[MORPHOLOGY]", generated_text, flags=re.IGNORECASE)
            captions[real_idx] = generated_text

    df['generated_caption'] = captions
    output_target = "test_" + OUTPUT_PATH if args.test_mode else OUTPUT_PATH
    df.to_csv(output_target, index=False)
    print(f"\nDone! Saved to {output_target}")
    
    if args.test_mode:
        print("\n=== PREVIEW OF CAPTIONS ===")
        for i in range(len(df)):
            print(f"[{df.iloc[i]['Majority Vote Label']}] {df.iloc[i]['generated_caption']}")

if __name__ == "__main__":
    main()
