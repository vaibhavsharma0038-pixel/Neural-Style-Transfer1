"""
train_fast.py — Enhanced Training Script for AdaIN Decoder
Supports:
  - Higher resolution (256x256)
  - Auto-resume from existing adain_decoder.pth checkpoint
  - Learning rate scheduling & gradient clipping
  - Live progress display & best-loss tracking
"""

import os
import sys
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from adain_model import VGGEncoder, Decoder
from utils import adain, calc_mean_std

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE     = 256          # Upgraded to 256x256 for sharper details
BATCH_SIZE     = 1
EPOCHS         = 15           # 15 fine-tuning epochs at high resolution
LR             = 5e-5         # Lower LR for fine-tuning without destabilizing weights
CONTENT_WEIGHT = 1.0
STYLE_WEIGHT   = 10.0
SAVE_PATH      = "adain_decoder.pth"
CONTENT_DIR    = "data/content"
STYLE_DIR      = "data/style"

print("=" * 60)
print(f"🚀 Neural Style Transfer — Model Fine-Tuning")
print(f"   Device        : {DEVICE}")
print(f"   Resolution    : {IMAGE_SIZE}x{IMAGE_SIZE} (High-Res)")
print(f"   Target Epochs : {EPOCHS}")
print(f"   Learning Rate : {LR}")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# Dataset with Data Augmentation
# ─────────────────────────────────────────────────────────────────────────────
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE + 32),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

class PairedStyleDataset(Dataset):
    def __init__(self, content_dir, style_dir, transform):
        self.content = [
            os.path.join(content_dir, f)
            for f in os.listdir(content_dir)
            if f.lower().endswith(IMG_EXTS)
        ]
        self.style = [
            os.path.join(style_dir, f)
            for f in os.listdir(style_dir)
            if f.lower().endswith(IMG_EXTS)
        ]
        self.transform = transform
        assert len(self.content) > 0, f"No content images found in {content_dir}"
        assert len(self.style) > 0,   f"No style images found in {style_dir}"

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        c = Image.open(self.content[idx]).convert("RGB")
        s = Image.open(random.choice(self.style)).convert("RGB")
        return self.transform(c), self.transform(s)

dataset    = PairedStyleDataset(CONTENT_DIR, STYLE_DIR, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, pin_memory=False)

print(f"📂 Dataset loaded:")
print(f"   Content Images : {len(dataset.content)}")
print(f"   Style Images   : {len(dataset.style)}")

# ─────────────────────────────────────────────────────────────────────────────
# Models & Checkpoint Resuming
# ─────────────────────────────────────────────────────────────────────────────
encoder = VGGEncoder().to(DEVICE).eval()
decoder = Decoder().to(DEVICE)

# Resume from existing trained model if available
if os.path.exists(SAVE_PATH):
    try:
        decoder.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE, weights_only=False))
        print(f"✅ Found existing checkpoint '{SAVE_PATH}'! Continuing training from here.")
    except Exception as e:
        print(f"⚠️ Could not load existing checkpoint ({e}). Starting fresh.")
else:
    print(f"ℹ️ No previous checkpoint found. Training from scratch.")

optimizer = optim.Adam(decoder.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# ─────────────────────────────────────────────────────────────────────────────
# Loss helpers
# ─────────────────────────────────────────────────────────────────────────────
def gram_matrix(feat):
    b, c, h, w = feat.size()
    f = feat.view(b, c, h * w)
    return f.bmm(f.transpose(1, 2)) / (c * h * w)

def content_loss(output_feat, target_feat):
    return nn.functional.mse_loss(output_feat, target_feat)

def style_loss(output_all, style_all):
    loss = 0
    for f_out, f_sty in zip(output_all, style_all):
        loss += nn.functional.mse_loss(gram_matrix(f_out), gram_matrix(f_sty))
    return loss

# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
print("\n🏋️ Training started...\n")
best_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    epoch_loss   = 0.0
    epoch_c_loss = 0.0
    epoch_s_loss = 0.0
    t0 = time.time()

    for batch_idx, (content_imgs, style_imgs) in enumerate(dataloader):
        content_imgs = content_imgs.to(DEVICE)
        style_imgs   = style_imgs.to(DEVICE)

        # ── Encode
        c_feat = encoder(content_imgs)
        s_feat = encoder(style_imgs)

        # ── AdaIN
        t = adain(c_feat, s_feat)

        # ── Decode
        output = decoder(t)
        output = torch.clamp(output, -3, 3)

        # ── Re-encode output for loss
        out_all   = encoder(output, return_all=True)
        style_all = encoder(style_imgs, return_all=True)

        c_loss = content_loss(out_all[3], t.detach())
        s_loss = style_loss(out_all, style_all)
        loss   = CONTENT_WEIGHT * c_loss + STYLE_WEIGHT * s_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        optimizer.step()

        epoch_loss   += loss.item()
        epoch_c_loss += c_loss.item()
        epoch_s_loss += s_loss.item()

        if (batch_idx + 1) % 12 == 0 or (batch_idx + 1) == len(dataloader):
            print(
                f"  Epoch {epoch:02d}/{EPOCHS} [{batch_idx+1:02d}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f} (Content: {c_loss.item():.4f} | Style: {s_loss.item():.4f})",
                flush=True,
            )

    scheduler.step()
    elapsed = time.time() - t0
    avg = epoch_loss / len(dataloader)
    print(
        f"\n✅ Epoch {epoch:02d}/{EPOCHS} complete | Avg Loss: {avg:.4f} | Elapsed: {elapsed:.0f}s",
        flush=True,
    )

    if avg < best_loss:
        best_loss = avg
        torch.save(decoder.state_dict(), SAVE_PATH)
        print(f"  💾 Checkpoint updated → {SAVE_PATH} (New Best Loss: {best_loss:.4f})\n", flush=True)

print("=" * 60)
print(f"🎉 Training complete! Best Loss: {best_loss:.4f}")
print(f"📦 Model saved to: {SAVE_PATH}")
print("=" * 60)
