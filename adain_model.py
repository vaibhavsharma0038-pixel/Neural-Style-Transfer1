"""
adain_model.py — VGG Encoder + AdaIN Decoder for fast Neural Style Transfer

Architecture:
  Encoder : VGG-19 up to relu4_1  (pretrained, frozen)
  AdaIN   : align content features to style statistics
  Decoder : learned mirror of the encoder
"""
import torch
import torch.nn as nn
import torchvision.models as models


# ─────────────────────────────────────────────────────────────────────────────
# VGG-19 Encoder  (outputs relu1_1, relu2_1, relu3_1, relu4_1)
# ─────────────────────────────────────────────────────────────────────────────
class VGGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features

        # relu4_1 is at index 20 in VGG-19 features
        self.slice1 = nn.Sequential(*list(vgg.children())[:2])    # relu1_1
        self.slice2 = nn.Sequential(*list(vgg.children())[:7])    # relu2_1
        self.slice3 = nn.Sequential(*list(vgg.children())[:12])   # relu3_1
        self.slice4 = nn.Sequential(*list(vgg.children())[:21])   # relu4_1

        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x, return_all=False):
        h1 = self.slice1(x)
        h2 = self.slice2(x)
        h3 = self.slice3(x)
        h4 = self.slice4(x)
        if return_all:
            return h1, h2, h3, h4
        return h4   # main encoding for AdaIN


# ─────────────────────────────────────────────────────────────────────────────
# Decoder  (learnable, mirrors the VGG-19 encoder)
# ─────────────────────────────────────────────────────────────────────────────
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # --- from relu4_1 (512 ch) → relu3_1 level (256 ch) ---
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, 3), nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3), nn.ReLU(inplace=True),

            # --- relu3_1 level → relu2_1 level (128 ch) ---
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, 3), nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, 3), nn.ReLU(inplace=True),

            # --- relu2_1 level → relu1_1 level (64 ch) ---
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, 3), nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3), nn.ReLU(inplace=True),

            # --- output: 3 ch image ---
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, 3),
        )

    def forward(self, x):
        return self.net(x)
