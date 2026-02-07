# CSIRO Biomass Prediction

My first Kaggle competition — built to gain hands-on experience with end-to-end ML pipelines. Finished top 20% (722/3803).

## Overview

Predicting dry biomass components (green, dead, clover) from agricultural field imagery. The competition used a weighted R² metric across 5 targets.

## Approach

**Dual-branch ensemble** combining two vision transformer pipelines:

### 1. SigLIP Branch
- Patch-based embeddings (520×520, 16px overlap) from Google's SigLIP vision-language model
- CLIP-style semantic features via text probing against domain anchors ("lush green pasture", "dry brown dead grass", etc.)
- Supervised feature engineering (PCA + PLS + GMM)
- 4-model GBM stack: CatBoost, LightGBM, HistGBM, GradientBoosting

### 2. DINOv3 Branch
- Fine-tuned `vit_large_patch16_dinov3` backbone
- FiLM (Feature-wise Linear Modulation) for three-view fusion (full + left/right image halves)
- Image preprocessing: bottom crop, orange date stamp inpainting
- A100-optimized training: bfloat16, OneCycleLR, EMA, stratified 5-fold CV

### Ensemble
- Per-target weighted fusion (65% DINO / 35% SigLIP)
- Post-processing with biological mass-balance constraints

## Key Techniques

- Vision-language embeddings (SigLIP)
- Feature-wise Linear Modulation (FiLM)
- Text-based semantic probing
- Multi-view image fusion
- Stratified cross-validation with custom binning
- Test-time augmentation
- Domain constraint enforcement

## Acknowledgments

Competition hosted by CSIRO on Kaggle. Built for learning purposes to familiarize myself with Kaggle pipelines and vision transformer fine-tuning.