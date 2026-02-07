"""
CSIRO BIOMASS - DINOV3 NN TRAINING (A100 OPTIMIZED)
This combines a massive transformer model with specific geometric logic (splitting the image) and runs it using the 
latest hardware acceleration techniques to predict biomass weights (Green, Dead, Clover, etc.) from pasture images.
This file was run on google colab using A100 GPU 40GB VRAM
"""

import os
import gc
import math
import random
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import timm
from timm.utils import ModelEmaV2

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import StratifiedKFold

import warnings
warnings.filterwarnings("ignore")

# A100 SPEED OPTIMIZATIONS - enable tf32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
#This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


class Config:
    DATA_ROOT = "/content/drive/MyDrive/csiro-biomass"
    TRAIN_CSV = "/content/drive/MyDrive/csiro-biomass/train.csv"
    OUT_DIR = "/content/drive/MyDrive/artifacts_nn_dinov3_a100"

    # MODEL
    MODEL_NAME = "vit_large_patch16_dinov3"
    IMG_SIZE = 512
    DROPOUT = 0.15  # Slightly lower dropout

    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    N_FOLDS = 5
    FOLDS_TO_TRAIN = [0, 1, 2, 3, 4]

    EPOCHS = 40             
    BATCH_SIZE = 8          
    GRAD_ACC = 2             
    NUM_WORKERS = 4

    LR_BACKBONE = 3e-5       # Slightly lower for stability
    LR_HEAD = 8e-4
    WEIGHT_DECAY = 1e-2
    WARMUP_RATIO = 0.1

    EMA_DECAY = 0.998        # Slower EMA for stability
    PATIENCE = 5             
    CLIP_GRAD_NORM = 1.0

    AMP = True
    AMP_DTYPE = torch.bfloat16

    # torch.compile for extra speed
    USE_COMPILE = False

    ALL_TARGET_COLS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
    PRED_COLS = ["Dry_Green_g", "Dry_Clover_g", "Dry_Dead_g"]

    R2_WEIGHTS = np.array([0.1, 0.1, 0.1, 0.2, 0.5], dtype=np.float32)

    # Target max for normalization
    TARGET_MAX = {
        "Dry_Clover_g": 71.7865,
        "Dry_Dead_g": 83.8407,
        "Dry_Green_g": 157.9836,
        "Dry_Total_g": 185.70,
        "GDM_g": 157.9836,
    }


cfg = Config()
os.makedirs(cfg.OUT_DIR, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(cfg.SEED)


# METRICS
def r2_score_np(y_true, y_pred):
  """
  Calculates R2 for a single column
  """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def weighted_r2_score(y_true, y_pred, weights):
  """
  Calculates the weighted average of all columns to give a single "Competition metric"
  """
    per = np.array([r2_score_np(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])], dtype=np.float32)
    w = weights.astype(np.float32)
    return float((per * w).sum() / w.sum()), per


# AUGMENTATIONS
def get_train_transforms():
    return A.Compose([
        A.RandomResizedCrop(
            size=(cfg.IMG_SIZE, cfg.IMG_SIZE),
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.06,
            scale_limit=0.12,
            rotate_limit=15,
            p=0.5,
            border_mode=cv2.BORDER_REFLECT_101
        ),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.15),
        A.OneOf([
            A.GaussNoise(var_limit=(5.0, 30.0), p=1.0),
            A.ISONoise(p=1.0),
        ], p=0.1),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.4),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms():
    return A.Compose([
        A.Resize(height=cfg.IMG_SIZE, width=cfg.IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# IMAGE CLEANING
def clean_image(img):
    """Remove bottom 10% and inpaint orange date stamps."""
    h, w = img.shape[:2]
    img = img[0:int(h * 0.90), :]

    # Inpaint orange date stamps
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([5, 150, 150]), np.array([25, 255, 255]))
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    if np.sum(mask) > 0:
        img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

    return img


# DATASET
class BiomassDataset(Dataset):
    def __init__(self, df, data_root, transform):
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.transform = transform
        self.paths = self.df["image_path"].values
        self.labels = self.df[cfg.ALL_TARGET_COLS].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def _read_image(self, rel_path):
        abs_path = rel_path if os.path.isabs(rel_path) else os.path.join(self.data_root, rel_path)
        img = cv2.imread(abs_path)
        if img is None:
            img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx):
        rel_path = self.paths[idx]
        img = self._read_image(rel_path)
        img = clean_image(img)

        h, w, _ = img.shape
        mid = w // 2
        left = img[:, :mid]
        right = img[:, mid:]
        full = img

        # Apply transforms
        left = self.transform(image=left)["image"]
        right = self.transform(image=right)["image"]
        full = self.transform(image=full)["image"]

        label = torch.from_numpy(self.labels[idx])
        return full, left, right, label

# MODEL
class FiLM(nn.Module):
    """Feature-wise Linear Modulation."""
    def __init__(self, feat_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2), #take the input vector and shrink it down to summarize the most important global info and ignore the noise.
            nn.GELU(),  # GELU slightly better than ReLU
            nn.Linear(feat_dim // 2, feat_dim * 2), #takes the compressed vector and blows it up to include gamma and beta vector.
        )

    def forward(self, context):
        gamma_beta = self.mlp(context)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1) #split the tensor into two pieces to derive gamma and beta
        return gamma, beta


class CSIROModelRegressor(nn.Module):
    """
    DINOv3 backbone + FiLM conditioning.
    Uses full + left + right views.

                  [ INPUT IMAGES ]
                 /       |        \
           (Full)      (Left)     (Right)
             |           |           |
             v           v           v
         DINOv3        DINOv3      DINOv3      <-- "The Backbone"
        (Shared)      (Shared)    (Shared)         (Extracts raw features)
             |           |           |
             v           v           v
        [Full Feat]  [Left Feat] [Right Feat]
             |           |           |
      --------           |           |
      v                  v           v
 |  FiLM  |------> | Modulate  | | Modulate  |   <-- "The Adapter"
 | Module | (γ,β)  | (Left * γ | |(Right * γ |       (Full view adjusts
                   |   + β)    | |   + β)    |        the zoomed views)
      |                  |           |
      |                  |           |
      v                  v           v
            CONCATENATION (Glued together)   
                         |
                         v
                [ Combined Vector ]
                 (Size: nf * 3)
                         |
                         v
                     The HEAD                    <-- "The Decision Maker"
                    (Layers 1-3)                     (Compresses data to 3 #'s)
                         |
                         v
                  [ Raw Outputs ]
               (Green, Clover, Dead)
    """

    def __init__(self, model_name, dropout=0.2, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        nf = self.backbone.num_features

        # # Enable gradient checkpointing for memory efficiency
        # if hasattr(self.backbone, 'set_grad_checkpointing'):
        #     self.backbone.set_grad_checkpointing(enable=True)

        self.film = FiLM(nf)

        # Larger head for better capacity
        self.head = nn.Sequential(
            nn.Linear(nf * 3, 512),   #input tensor is huge becuase we concatenated 3 views (full, left, right) earlier
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )
        #positivity enforcer. standard NN can output negative numbers but biomass can't be negative
        #Softplus is a mathematical function that converts any number into a positive number smoothly.
        self.softplus = nn.Softplus(beta=1.0)

        # pytorch trick to store numbers inside a model that should not be learned
        self.register_buffer('max_green', torch.tensor(cfg.TARGET_MAX["Dry_Green_g"]))
        self.register_buffer('max_clover', torch.tensor(cfg.TARGET_MAX["Dry_Clover_g"]))
        self.register_buffer('max_dead', torch.tensor(cfg.TARGET_MAX["Dry_Dead_g"]))

    def forward(self, full_img, left_img, right_img):
        full_feat = self.backbone(full_img)
        left_feat = self.backbone(left_img)
        right_feat = self.backbone(right_img)

        # FiLM modulation from full image context - forces local views to agree with the global views
        gamma, beta = self.film(full_feat)
        left_mod = left_feat * (1.0 + gamma) + beta   # +1.0 creates a residual connection
        right_mod = right_feat * (1.0 + gamma) + beta

        comb = torch.cat([full_feat, left_mod, right_mod], dim=1)

        prim_norm = self.softplus(self.head(comb))
        green_norm = prim_norm[:, 0:1]
        clover_norm = prim_norm[:, 1:2]
        dead_norm = prim_norm[:, 2:3]

        # Convert to raw values
        # the ground truth label was normalized using - actual grams in image i / global max. so the model outputs estimation for this fraction.
        # thats why to convert back to raw values we multiply by global max
        green = green_norm * self.max_green
        clover = clover_norm * self.max_clover
        dead = dead_norm * self.max_dead

        gdm = green + clover
        total = gdm + dead

        packed = torch.cat([green, dead, clover, gdm, total], dim=1)
        return packed

# LOSS
class BiomassLoss(nn.Module):
    """Weighted normalized loss with Huber."""
    def __init__(self):
        super().__init__()

        #load max possible values for all targets
        self.register_buffer('max_vec', torch.tensor([
            cfg.TARGET_MAX["Dry_Green_g"],
            cfg.TARGET_MAX["Dry_Dead_g"],
            cfg.TARGET_MAX["Dry_Clover_g"],
            cfg.TARGET_MAX["GDM_g"],
            cfg.TARGET_MAX["Dry_Total_g"],
        ]))
        self.register_buffer('weights', torch.tensor(cfg.R2_WEIGHTS))

        self.huber = nn.SmoothL1Loss(beta=1.0, reduction="none")

    def forward(self, preds, labels):
        preds_norm = preds / self.max_vec
        labels_norm = labels / self.max_vec

        #It calculates the error for each of the 5 targets separately.
        per = self.huber(preds_norm, labels_norm).mean(dim=0)
        w = self.weights / self.weights.sum()
        return (per * w).sum() #multiplies each error by its importance and sums it up


# OPTIMIZER & SCHEDULER
def build_optimizer(model):
    backbone_params = list(model.backbone.parameters())
    head_params = [p for n, p in model.named_parameters() if "backbone" not in n]

    return optim.AdamW([
        {"params": backbone_params, "lr": cfg.LR_BACKBONE, "weight_decay": cfg.WEIGHT_DECAY},
        {"params": head_params, "lr": cfg.LR_HEAD, "weight_decay": cfg.WEIGHT_DECAY},
    ])


def build_scheduler(optimizer, steps_per_epoch):
    total_steps = cfg.EPOCHS * steps_per_epoch

    return optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[cfg.LR_BACKBONE, cfg.LR_HEAD],
        total_steps=total_steps,
        pct_start=cfg.WARMUP_RATIO,   #Starts with a tiny learning rate and increases it for the first ~10% of training. 
        anneal_strategy='cos',        #sets up how the lr drops from max to zero.
        div_factor=25,            #sets up the initial lerning rate. formula - max_lr/div_factor
        final_div_factor=1000,    #this sets the final lr. formula - max_lr/1000
    )

# TRAINING FUNCTIONS
def train_epoch(model, loader, optimizer, scheduler, loss_fn, ema=None):
    model.train()
    running = 0.0
    n_samples = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for full, left, right, labels in pbar:
        # non_blocking=True: A speed hack. It allows the CPU to prepare the next batch of data while the 
        # GPU is currently busy moving this batch into VRAM. It creates an asynchronous pipeline.
        full = full.to(cfg.DEVICE, non_blocking=True)
        left = left.to(cfg.DEVICE, non_blocking=True)
        right = right.to(cfg.DEVICE, non_blocking=True)
        labels = labels.to(cfg.DEVICE, non_blocking=True)

        #set_to_none=True: Instead of setting gradients to 0 (which takes memory write time), it sets the pointer to None. 
        #This is slightly faster and saves memory.
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast("cuda", dtype=cfg.AMP_DTYPE, enabled=cfg.AMP):
            preds = model(full, left, right)
            loss = loss_fn(preds, labels)

        loss.backward()
        #clip_grad_norm_: Safety Mechanism.
        #If a gradient is massive (e.g., 1000.0), it chops it down to a max limit (e.g., 1.0).
        #Why? Prevents "Exploding Gradients" which can destroy a model's weights in a single batch.
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP_GRAD_NORM)
        optimizer.step()
        scheduler.step()

        if ema is not None:
            ema.update(model)

        running += loss.item() * labels.size(0) #loss is average over the batch. so we multiply by the batch size to get the count
        n_samples += labels.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
    #true avg loss for the entire dataset.
    return running / n_samples


@torch.no_grad()
def validate(model, loader, loss_fn):
    model.eval()
    all_preds = []
    all_labels = []
    running = 0.0
    n_samples = 0

    for full, left, right, labels in tqdm(loader, desc="Val", leave=False):
        full = full.to(cfg.DEVICE, non_blocking=True)
        left = left.to(cfg.DEVICE, non_blocking=True)
        right = right.to(cfg.DEVICE, non_blocking=True)
        labels = labels.to(cfg.DEVICE, non_blocking=True)

        with torch.autocast("cuda", dtype=cfg.AMP_DTYPE, enabled=cfg.AMP):
            preds = model(full, left, right)
            loss = loss_fn(preds, labels)

        running += loss.item() * labels.size(0)
        n_samples += labels.size(0)

        all_preds.append(preds.float().cpu().numpy())
        all_labels.append(labels.float().cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    avg_loss = running / n_samples
    wr2, per = weighted_r2_score(all_labels, all_preds, cfg.R2_WEIGHTS)
    return avg_loss, wr2, per, all_preds

# STRATIFIED FOLDS
def create_stratified_folds(df, n_splits=5, seed=42):
    df = df.copy().reset_index(drop=True)

    df["bin_total"] = pd.qcut(df["Dry_Total_g"], q=3, labels=["L", "M", "H"])
    living_mass = df["Dry_Clover_g"] + df["Dry_Green_g"]
    df["clover_frac"] = df["Dry_Clover_g"] / (living_mass + 1e-6)
    df["bin_clover"] = pd.cut(df["clover_frac"], bins=[-0.1, 0.2, 1.1], labels=["Lo", "Hi"])
    df["state_key"] = df["State"].astype(str)

    df["key_L1"] = df["state_key"] + "_" + df["bin_total"].astype(str) + "_" + df["bin_clover"].astype(str)
    df["key_L2"] = df["state_key"] + "_" + df["bin_total"].astype(str)
    df["key_L3"] = df["state_key"]

    df["final_stratify"] = df["key_L1"]

    # Collapse rare keys
    for fallback in ["key_L2", "key_L3"]:
        counts = df["final_stratify"].value_counts()
        rare = counts[counts < n_splits].index
        df.loc[df["final_stratify"].isin(rare), "final_stratify"] = df.loc[df["final_stratify"].isin(rare), fallback]

    counts = df["final_stratify"].value_counts()
    rare = counts[counts < n_splits].index
    if len(rare) > 0:
        df.loc[df["final_stratify"].isin(rare), "final_stratify"] = "Misc"

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    df["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(df, df["final_stratify"])):
        df.loc[val_idx, "fold"] = fold

    return df


if __name__ == "__main__":
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name}")
        print(f"Memory: {gpu_mem:.1f} GB")

    print("=" * 80)
    print("DINOV3 NN TRAINING (A100 OPTIMIZED)")
    print("=" * 80)
    print(f"Device: {cfg.DEVICE}")
    print(f"Model:  {cfg.MODEL_NAME}")
    print(f"IMG:    {cfg.IMG_SIZE}")
    print(f"Batch:  {cfg.BATCH_SIZE}")
    print(f"Epochs: {cfg.EPOCHS}")
    print(f"AMP:    {cfg.AMP} ({cfg.AMP_DTYPE})")
    print(f"Compile: {cfg.USE_COMPILE}")

    # Load data
    print("\n[1] Loading data...")
    train_long = pd.read_csv(cfg.TRAIN_CSV)

    train_df = train_long.pivot_table(
        index=["image_path", "Sampling_Date", "State", "Species", "Pre_GSHH_NDVI", "Height_Ave_cm"],
        columns="target_name",
        values="target",
        aggfunc="first",
    ).reset_index()


    train_df = create_stratified_folds(train_df, n_splits=cfg.N_FOLDS, seed=cfg.SEED)

    print(f"Training samples: {len(train_df)}")
    print(f"\nFold distribution:\n{train_df['fold'].value_counts().sort_index()}")

    # Training
    print("\n[2] Starting CV training...")

    oof_preds = np.zeros((len(train_df), len(cfg.ALL_TARGET_COLS)), dtype=np.float32)
    oof_labels = train_df[cfg.ALL_TARGET_COLS].values.astype(np.float32)
    fold_scores = []

    for fold in cfg.FOLDS_TO_TRAIN:
        print(f"\n{'=' * 70}")
        print(f"FOLD {fold + 1}/{cfg.N_FOLDS}")
        print(f"{'=' * 70}")

        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()

        tr_df = train_df[train_df["fold"] != fold].reset_index(drop=True)
        va_df = train_df[train_df["fold"] == fold].reset_index(drop=True)
        print(f"Train: {len(tr_df)} | Val: {len(va_df)}")

        # Datasets
        train_ds = BiomassDataset(tr_df, cfg.DATA_ROOT, get_train_transforms())
        val_ds = BiomassDataset(va_df, cfg.DATA_ROOT, get_val_transforms())

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.BATCH_SIZE,
            shuffle=True,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=True,              #speeds up cpu to gpu transfer singificantly
            drop_last=True,
            persistent_workers=(cfg.NUM_WORKERS > 0),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.BATCH_SIZE * 2,
            shuffle=False,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
            persistent_workers=(cfg.NUM_WORKERS > 0),
        )

        # Model
        model = CSIROModelRegressor(cfg.MODEL_NAME, dropout=cfg.DROPOUT, pretrained=True).to(cfg.DEVICE)

        # torch.compile for A100
        if cfg.USE_COMPILE and hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode="reduce-overhead")
                print("Model compiled with torch.compile")
            except Exception as e:
                print(f"torch.compile failed: {e}")

        # Optimizer, scheduler, loss
        optimizer = build_optimizer(model)
        steps_per_epoch = len(train_loader)
        scheduler = build_scheduler(optimizer, steps_per_epoch)
        loss_fn = BiomassLoss().to(cfg.DEVICE)

        # EMA
        # Original Model: Optimizes for the Current Batch (Noisy). EMA Model: Optimizes for the Long-Term Trend (Stable).
        #Weight_EMA = (0.999 * Weight_Old) + (0.001 * Weight_Current)
        ema = ModelEmaV2(model, decay=cfg.EMA_DECAY)

        best_r2 = -1e9
        best_preds = None
        patience = 0
        save_path = os.path.join(cfg.OUT_DIR, f"best_model_fold{fold}.pth")

        for epoch in range(1, cfg.EPOCHS + 1):
            print(f"\nEpoch {epoch}/{cfg.EPOCHS}")

            tr_loss = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, ema=ema)
            #We use EMA for validation because it represents the stable, generalizable intelligence our system, stripping away the random noise of the final training step.
            va_loss, va_r2, per_r2, va_preds = validate(ema.module, val_loader, loss_fn)

            # Format per-target R2
            per_str = " | ".join([f"{cfg.ALL_TARGET_COLS[i][:5]}: {per_r2[i]:.3f}" for i in range(len(per_r2))])

            is_best = va_r2 > best_r2
            print(f"  TrLoss: {tr_loss:.5f} | VaLoss: {va_loss:.5f}")
            print(f"  Weighted R²: {va_r2:.4f} {'[BEST]' if is_best else ''}")
            print(f"  {per_str}")

            if is_best:
                best_r2 = va_r2
                best_preds = va_preds.copy()
                patience = 0
                torch.save(ema.module.state_dict(), save_path)
                print(f"  → Saved: {save_path}")
            else:
                patience += 1
                if patience >= cfg.PATIENCE:
                    print(f"  → Early stopping (no improvement for {cfg.PATIENCE} epochs)")
                    break

        # Store OOF predictions
        val_indices = train_df[train_df["fold"] == fold].index.values
        oof_preds[val_indices] = best_preds
        fold_scores.append(best_r2)

        print(f"\nFold {fold} best Weighted R²: {best_r2:.4f}")

        # Cleanup
        del model, optimizer, scheduler, ema, train_loader, val_loader, train_ds, val_ds, loss_fn
        gc.collect()
        torch.cuda.empty_cache()

    # Final OOF evaluation
    print("\n" + "=" * 80)
    print("FINAL OOF EVALUATION")
    print("=" * 80)

    final_r2, final_per = weighted_r2_score(oof_labels, oof_preds, cfg.R2_WEIGHTS)
    print(f"\nOOF Weighted R²: {final_r2:.4f}")
    print("\nPer-target R²:")
    for i, t in enumerate(cfg.ALL_TARGET_COLS):
        print(f"  {t}: {final_per[i]:.4f}")

    print("\nFold Scores:")
    for i, s in enumerate(fold_scores):
        print(f"  Fold {i}: {s:.4f}")
    print(f"  Mean: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")

    # Save config
    import pickle
    config_dict = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
    with open(os.path.join(cfg.OUT_DIR, "config.pkl"), "wb") as f:
        pickle.dump(config_dict, f)

    # Save OOF predictions
    oof_df = train_df[["image_path", "fold"]].copy()
    for i, col in enumerate(cfg.ALL_TARGET_COLS):
        oof_df[f"pred_{col}"] = oof_preds[:, i]
    oof_df.to_csv(os.path.join(cfg.OUT_DIR, "oof_predictions.csv"), index=False)

    print(f"\nModels saved to: {cfg.OUT_DIR}")
    print("DONE.")