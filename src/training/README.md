# Training

This package holds the training pipeline for 2D pretraining, 2D diffusion, and 3D fine-tuning.

Phases:
1) Build JSONL manifests for slices and volumes.
2) Train 2D AutoencoderKL.
3) Train 2D DiffusionModelUNet in latent space.
4) Inflate to 3D and fine-tune on volumes.
