import os
import argparse
import random
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
ANNOTATIONS_PATH = "quilt_data/annotations.csv"
IMAGES_DIR = "images"
MODEL_NAME = "hf-hub:wisdomik/QuiltNet-B-32"
LABEL_MAP = {"HP": 1, "SSA": 0}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Dataset Class ---
class MHISTImageDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, clip_preprocess=None, 
                 use_stain_norm=True, normalizer=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.clip_preprocess = clip_preprocess
        self.use_stain_norm = use_stain_norm
        self.normalizer = normalizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['Image Name'])
        image = Image.open(img_path).convert("RGB")
        
        if self.use_stain_norm and self.normalizer:
            try:
                img_tensor = transforms.ToTensor()(image)
                image_norm, _, _ = self.normalizer.normalize(I=img_tensor)
                image = transforms.ToPILImage()(image_norm)
            except Exception:
                pass 
        
        if self.transform:
            image = self.transform(image)
        if self.clip_preprocess:
            image = self.clip_preprocess(image)
            
        label = row['Majority Vote Label']
        return image, label

# --- 2. Image-Only Classifier ---
class ImageOnlyQuiltClassifier(nn.Module):
    def __init__(self, base_model, num_classes=2, dropout_rate=0.4):
        super().__init__()
        self.base_model = base_model
        self.fc1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU()
        )
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(5)])
        self.out = nn.Linear(256, num_classes)
        
    def forward(self, image):
        # We only use the image encoder
        features = self.base_model.encode_image(image)
        features = features / features.norm(dim=-1, keepdim=True)
        
        x = self.fc1(features.float())
        
        if self.training:
            logits = sum([self.out(drop(x)) for drop in self.dropouts]) / len(self.dropouts)
        else:
            logits = self.out(x)
        return logits

# --- 3. Loss & Utilities ---
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
        state_dict = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()
        torch.save(state_dict, self.path)

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

# --- 4. Main Execution Logic ---
def run_experiment(num_trials=50, freeze_backbone=False, results_base_dir="results"):
    mode_str = "frozen" if freeze_backbone else "finetuned"
    experiment_name = "image_only"
    
    BASE_DIR = os.path.join(results_base_dir, "image_only", mode_str)
    TRIALS_DIR = os.path.join(BASE_DIR, "trials")
    os.makedirs(TRIALS_DIR, exist_ok=True)
    
    print(f"\n--- Starting IMAGE_ONLY {mode_str.upper()} Baseline Experiment ---")
    print(f"Results directory: {BASE_DIR}")
    
    # Load data
    df = pd.read_csv(ANNOTATIONS_PATH)
    
    # Split
    test_df = df[df['Partition'] == 'test'].reset_index(drop=True)
    other_df = df[df['Partition'] == 'train'].reset_index(drop=True)
    train_df, val_df = train_test_split(
        other_df, test_size=0.1, stratify=other_df['Majority Vote Label'], random_state=42
    )
    
    # Stain Normalization
    print("Fitting Macenko Stain Normalizer...")
    target_img_path = os.path.join(IMAGES_DIR, df.iloc[0]['Image Name'])
    target_img = Image.open(target_img_path).convert("RGB")
    normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
    normalizer.fit(transforms.ToTensor()(target_img))
    
    # Load Base Model
    print(f"Loading base model: {MODEL_NAME}...")
    base_model, _, clip_preprocess = open_clip.create_model_and_transforms(MODEL_NAME)
    base_model = base_model.to(device)
    
    # Freeze logic
    if freeze_backbone:
        print("Freezing backbone parameters...")
        for param in base_model.parameters():
            param.requires_grad = False
            
    best_study_score = -float('inf')
    
    def objective(trial):
        nonlocal best_study_score
        head_lr = trial.suggest_float("head_lr", 1e-4, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        dropout_rate = trial.suggest_float("dropout_rate", 0.3, 0.6)
        focal_alpha = trial.suggest_float("focal_alpha", 0.2, 0.5)
        focal_gamma = trial.suggest_float("focal_gamma", 1.5, 3.0)
        batch_size = trial.suggest_categorical("batch_size", [16, 32])
        
        if not freeze_backbone:
            lr_ratio = trial.suggest_float("lr_ratio", 0.01, 0.1) 
            backbone_lr = head_lr * lr_ratio
        else:
            backbone_lr = 0.0

        train_dataset = MHISTImageDataset(train_df, IMAGES_DIR, transform=histology_transform(True), clip_preprocess=clip_preprocess, normalizer=normalizer)
        val_dataset = MHISTImageDataset(val_df, IMAGES_DIR, transform=histology_transform(False), clip_preprocess=clip_preprocess, normalizer=normalizer)
        
        count_dict = train_df['Majority Vote Label'].value_counts().to_dict()
        weights = [1.0 / count_dict.get(label, 1) for label in train_df['Majority Vote Label']]
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        classifier = ImageOnlyQuiltClassifier(base_model, dropout_rate=dropout_rate).to(device)
        try:
            classifier = torch.compile(classifier)
        except Exception:
            pass

        if freeze_backbone:
            param_groups = [{'params': [p for p in classifier.parameters() if p.requires_grad], 'lr': head_lr}]
        else:
            param_groups = [
                {'params': classifier.base_model.parameters(), 'lr': backbone_lr},
                {'params': [p for n, p in classifier.named_parameters() if 'base_model' not in n], 'lr': head_lr}
            ]
            
        optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma) 
        scaler = torch.amp.GradScaler('cuda')
        
        temp_trial_path = os.path.join(TRIALS_DIR, f'trial_temp_{trial.number}.pth')
        early_stopping = EarlyStopping(patience=5, path=temp_trial_path)
        
        for epoch in range(30):
            classifier.train()
            for imgs, lbls in train_loader:
                imgs = imgs.to(device)
                tgts = torch.tensor([LABEL_MAP[l] for l in lbls]).to(device)
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    outputs = classifier(imgs)
                    loss = criterion(outputs, tgts)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
            scheduler.step()
            
            classifier.eval()
            all_preds, all_tgts = [], []
            with torch.no_grad():
                for imgs, lbls in val_loader:
                    imgs = imgs.to(device)
                    with torch.amp.autocast('cuda'):
                        outputs = classifier(imgs)
                    preds = outputs.argmax(dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_tgts.extend([LABEL_MAP[l] for l in lbls])
            
            val_f1 = f1_score(all_tgts, all_preds, average='macro')
            trial.report(val_f1, epoch)
            if trial.should_prune():
                if os.path.exists(temp_trial_path): os.remove(temp_trial_path)
                raise optuna.exceptions.TrialPruned()
            
            early_stopping(val_f1, classifier)
            if early_stopping.early_stop:
                break
        
        if early_stopping.best_score > best_study_score:
            best_study_score = early_stopping.best_score
            final_best_path = os.path.join(BASE_DIR, 'best_study_model.pth')
            if os.path.exists(temp_trial_path):
                import shutil
                shutil.move(temp_trial_path, final_best_path)
                print(f"  *** New best study model saved (Mode: {mode_str}, Score: {best_study_score:.4f}) ***")
        else:
            if os.path.exists(temp_trial_path): os.remove(temp_trial_path)
                
        return early_stopping.best_score

    # Run Optuna Study
    study_name = f"mhist_image_only_{mode_str}"
    storage_path = os.path.join(BASE_DIR, f"optuna_{study_name}.db")
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=num_trials)
    
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best Val Macro-F1: {study.best_value}")
    
    # Final Evaluation
    print("\nEvaluating best model on Test set...")
    final_best_path = os.path.join(BASE_DIR, 'best_study_model.pth')
    
    best_model = ImageOnlyQuiltClassifier(base_model, dropout_rate=study.best_params['dropout_rate']).to(device)
    best_model.load_state_dict(torch.load(final_best_path))
    best_model.eval()
    
    test_dataset = MHISTImageDataset(test_df, IMAGES_DIR, transform=histology_transform(False), clip_preprocess=clip_preprocess, normalizer=normalizer)
    test_loader = DataLoader(test_dataset, batch_size=study.best_params['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
    
    all_preds, all_tgts = [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            outputs = best_model(imgs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_tgts.extend([LABEL_MAP[l] for l in lbls])
            
    report = classification_report(all_tgts, all_preds, target_names=[INV_LABEL_MAP[0], INV_LABEL_MAP[1]], digits=4)
    print("\nTest Set Classification Report:")
    print(report)
    
    with open(os.path.join(BASE_DIR, f"final_report_{study_name}.txt"), "w") as f:
        f.write(f"Experiment: IMAGE_ONLY {mode_str.upper()}\n")
        f.write(f"Best Params: {study.best_params}\n\n")
        f.write(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze the backbone visual encoder")
    parser.add_argument("--results_dir", type=str, default="results", help="Base directory for results")
    args = parser.parse_args()
    
    run_experiment(num_trials=args.trials, freeze_backbone=args.freeze_backbone, results_base_dir=args.results_dir)

