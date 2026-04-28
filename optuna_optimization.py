import os
import ast
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import optuna
import open_clip
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torchstain

# --- 0. Configuration & Global Setup ---
ANNOTATIONS_PATH = "annotations.csv"
IMAGES_DIR = "images"
MODEL_NAME = "hf-hub:wisdomik/QuiltNet-B-32"
LABEL_MAP = {"HP": 1, "SSA": 0}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 0.1 Create Trials Directory ---
TRIALS_DIR = "optuna_trials"
os.makedirs(TRIALS_DIR, exist_ok=True)

# --- 1. Prepare the Multimodal Data ---
df = pd.read_csv(ANNOTATIONS_PATH)
captions_df = pd.read_csv('mhist_with_captions.csv')
captions_df = captions_df.rename(columns={'image': 'Image Name'})
multimodal_df = df.merge(captions_df[['Image Name', 'top_prompts', 'top_scores']], on='Image Name', how='left')

# --- 2. Tripartite Split (Train, Val, Test) ---
def prepare_splits(multimodal_df):
    test_df = multimodal_df[multimodal_df['Partition'] == 'test'].reset_index(drop=True)
    other_df = multimodal_df[multimodal_df['Partition'] == 'train'].reset_index(drop=True)
    train_df, val_df = train_test_split(
        other_df, test_size=0.15, stratify=other_df['Majority Vote Label'], random_state=42
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

train_df, val_df, test_df = prepare_splits(multimodal_df)

# --- 3. Stain Normalization & Advanced Augmentation ---
print("Fitting Macenko Stain Normalizer...")
target_img_path = os.path.join(IMAGES_DIR, df.iloc[0]['Image Name'])
target_img = Image.open(target_img_path).convert("RGB")
normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
normalizer.fit(transforms.ToTensor()(target_img))

def histology_transform(is_train=True):
    aug_list = []
    if is_train:
        aug_list += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        ]
    return transforms.Compose(aug_list)

# --- 4. Dataset with Stain Normalization integration ---
class MultimodalMHISTDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, clip_preprocess=None, tokenizer=None, 
                 use_stain_norm=True, min_confidence=0.5):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.clip_preprocess = clip_preprocess
        self.tokenizer = tokenizer
        self.use_stain_norm = use_stain_norm
        self.min_confidence = min_confidence

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['Image Name'])
        image = Image.open(img_path).convert("RGB")
        if self.use_stain_norm:
            try:
                img_tensor = transforms.ToTensor()(image)
                image_norm, _, _ = normalizer.normalize(I=img_tensor)
                image = transforms.ToPILImage()(image_norm)
            except Exception:
                pass 
        if self.transform:
            image = self.transform(image)
        if self.clip_preprocess:
            image = self.clip_preprocess(image)
        prompts = ast.literal_eval(row['top_prompts'])
        scores = ast.literal_eval(row['top_scores'])
        relevant = [p for p, s in zip(prompts, scores) if s >= self.min_confidence]
        if not relevant: relevant = [prompts[0]]
        caption_text = ". ".join(relevant)
        text_tokens = self.tokenizer([str(caption_text)])[0]
        label = row['Majority Vote Label']
        return image, text_tokens, label

# --- 5. Gated Multimodal Unit (GMU) Implementation ---
class GatedMultimodalUnit(nn.Module):
    def __init__(self, img_dim, text_dim, hidden_dim):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.gate = nn.Linear(img_dim + text_dim, hidden_dim)
        self.activation = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, img_feat, text_feat):
        h_i = self.activation(self.img_proj(img_feat))
        h_t = self.activation(self.text_proj(text_feat))
        z = self.sigmoid(self.gate(torch.cat((img_feat, text_feat), dim=-1)))
        h = z * h_i + (1 - z) * h_t
        return h

# --- 6. Classifier with GMU and Multi-sample Dropout ---
class MultimodalQuiltClassifier(nn.Module):
    def __init__(self, base_model, num_classes=2, dropout_rate=0.4, freeze_backbone=False):
        super().__init__()
        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = not freeze_backbone
        self.gmu = GatedMultimodalUnit(512, 512, 512)
        self.fc1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU()
        )
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(5)])
        self.out = nn.Linear(256, num_classes)
        
    def forward(self, image, text):
        image_features = self.base_model.encode_image(image)
        text_features = self.base_model.encode_text(text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        combined = self.gmu(image_features.float(), text_features.float())
        x = self.fc1(combined)
        
        if self.training:
            logits = sum([self.out(drop(x)) for drop in self.dropouts]) / len(self.dropouts)
        else:
            logits = self.out(x)
        return logits

# --- 7. Custom Loss & Tools ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt)**self.gamma * ce_loss).mean()

class EarlyStopping:
    def __init__(self, patience=8, path='best_model.pth'):
        self.patience, self.path = patience, path
        self.counter, self.best_score, self.early_stop = 0, None, False

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        # Handle compiled models (PyTorch 2.0+)
        state_dict = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()
        torch.save(state_dict, self.path)

# --- 8. Load Base Model ---
print(f"Loading base model: {MODEL_NAME}...")
base_model, _, clip_preprocess = open_clip.create_model_and_transforms(MODEL_NAME)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)
base_model = base_model.to(device)

# --- 9. Optuna Objective (Performance Optimized) ---
def objective(trial):
    head_lr = trial.suggest_float("head_lr", 1e-4, 1e-2, log=True)
    lr_ratio = trial.suggest_float("lr_ratio", 0.01, 0.1) 
    backbone_lr = head_lr * lr_ratio
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.6)
    focal_alpha = trial.suggest_float("focal_alpha", 0.2, 0.5)
    focal_gamma = trial.suggest_float("focal_gamma", 1.0, 3.0)
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    min_confidence = trial.suggest_float("min_confidence", 0.5, 0.7)
    
    train_dataset = MultimodalMHISTDataset(train_df, IMAGES_DIR, transform=histology_transform(True), clip_preprocess=clip_preprocess, tokenizer=tokenizer, min_confidence=min_confidence)
    val_dataset = MultimodalMHISTDataset(val_df, IMAGES_DIR, transform=histology_transform(False), clip_preprocess=clip_preprocess, tokenizer=tokenizer, min_confidence=min_confidence)
    
    count_dict = train_df['Majority Vote Label'].value_counts().to_dict()
    weights = [1.0 / count_dict.get(label, 1) for label in train_df['Majority Vote Label']]
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    
    # PERFORMANCE: num_workers and pin_memory
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    classifier = MultimodalQuiltClassifier(base_model, dropout_rate=dropout_rate, freeze_backbone=False).to(device)
    
    # PERFORMANCE: torch.compile
    try:
        classifier = torch.compile(classifier)
    except Exception:
        pass

    param_groups = [
        {'params': classifier.base_model.parameters(), 'lr': backbone_lr},
        {'params': [p for n, p in classifier.named_parameters() if 'base_model' not in n], 'lr': head_lr}
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
    criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma) 
    
    # PERFORMANCE: Modern GradScaler (torch.amp)
    scaler = torch.amp.GradScaler('cuda')
    
    early_stopping = EarlyStopping(patience=5, path=os.path.join(TRIALS_DIR, f'temp_trial_{trial.number}.pth'))
    
    for epoch in range(40):
        classifier.train()
        for images, texts, labels in train_loader:
            images, texts = images.to(device), texts.to(device)
            targets = torch.tensor([LABEL_MAP[l] for l in labels]).to(device)
            optimizer.zero_grad()
            
            # PERFORMANCE: Modern Mixed Precision (torch.amp)
            with torch.amp.autocast('cuda'):
                outputs = classifier(images, texts)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        scheduler.step()
        classifier.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, texts, labels in val_loader:
                images, texts = images.to(device), texts.to(device)
                targets = torch.tensor([LABEL_MAP[l] for l in labels]).to(device)
                
                with torch.amp.autocast('cuda'):
                    outputs = classifier(images, texts)
                
                all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                all_labels.extend(targets.cpu().numpy())
        
        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        early_stopping(val_f1, classifier)
        if early_stopping.early_stop: break
        trial.report(val_f1, epoch)
        if trial.should_prune(): raise optuna.exceptions.TrialPruned()
            
    return early_stopping.best_score

# --- 10. Run Optimization (Direction: Maximize) ---
db_path = os.path.join(TRIALS_DIR, "mhist_optuna.db")
storage_url = f"sqlite:///{db_path}"
study = optuna.create_study(study_name="mhist_multimodal_v1", storage=storage_url, direction="maximize", load_if_exists=True)

# --- 10.1 Enqueue known good parameters as a 'hint' ---
study.enqueue_trial({
    'head_lr': 0.000174476653545221,
    'lr_ratio': 0.04386199793240394,
    'weight_decay': 0.009426625664495609,
    'dropout_rate': 0.5771301040951384,
    'focal_alpha': 0.2895124290436869,
    'focal_gamma': 2.2320584202656546,
    'batch_size': 32,
    'min_confidence': 0.5137964509552385
})

study.optimize(objective, n_trials=100)

print(f"\nBest Trial: {study.best_trial.number}")
print(f"Best Val Macro-F1: {study.best_value}")
print("Best Params:", study.best_params)

# --- 11. Save Best Model & Final Test ---
best_path = os.path.join(TRIALS_DIR, f'temp_trial_{study.best_trial.number}.pth')
best_model = MultimodalQuiltClassifier(base_model, dropout_rate=study.best_params['dropout_rate'], freeze_backbone=False).to(device)

# Load state dict (handle potential _orig_mod prefix from older trials if necessary)
state_dict = torch.load(best_path)
if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    
best_model.load_state_dict(state_dict)
torch.save(best_model.state_dict(), os.path.join(TRIALS_DIR, 'best_multimodal_classifier_final.pth'))

test_dataset = MultimodalMHISTDataset(test_df, IMAGES_DIR, transform=histology_transform(False), clip_preprocess=clip_preprocess, tokenizer=tokenizer, min_confidence=study.best_params['min_confidence'])
test_loader = DataLoader(test_dataset, batch_size=study.best_params['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

test_preds, test_targets = [], []
best_model.eval()
with torch.no_grad():
    for images, texts, labels in test_loader:
        images, texts = images.to(device), texts.to(device)
        targets = torch.tensor([LABEL_MAP[l] for l in labels]).to(device)
        with torch.amp.autocast('cuda'):
            outputs = best_model(images, texts)
        test_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        test_targets.extend(targets.cpu().numpy())

print("\nFinal Test Set Performance (Fully Optimized Model):")
print(classification_report(test_targets, test_preds, target_names=["SSA", "HP"], digits=4))
