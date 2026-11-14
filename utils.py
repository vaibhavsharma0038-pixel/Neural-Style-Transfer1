"""
utils.py — AdaIN helpers for Neural Style Transfer
"""
import torch


def calc_mean_std(feat, eps=1e-5):
    """Compute per-channel mean and std of a feature map (N, C, H, W)."""
    N, C = feat.size()[:2]
    feat_var  = feat.view(N, C, -1).var(dim=2) + eps
    feat_std  = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adain(content_feat, style_feat):
    """
    Adaptive Instance Normalisation:
      normalise content features, then re-scale to style statistics.
    """
    c_mean, c_std = calc_mean_std(content_feat)
    s_mean, s_std = calc_mean_std(style_feat)
    normalised = (content_feat - c_mean) / c_std
    return normalised * s_std + s_mean
