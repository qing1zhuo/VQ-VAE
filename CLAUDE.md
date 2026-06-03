# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- **Conda environment**: `dl` (Python ≥ 3.9, PyTorch ≥ 2.0 with CUDA)
- **VS Code** uses `D:\anaconda3\envs\dl\python.exe` as interpreter
- Verify CUDA: `python test.py`

## Project Architecture

This is a three-modality VQ-VAE implementation (van den Oord et al., 2017) with autoregressive priors. The code is organized into parallel image, audio, and video branches, each with independent Model/Train/GetData modules.

### Core design pattern

All modalities follow the same two-phase pipeline:

1. **Phase 1 — VQ-VAE training**: Encoder → Vector Quantization (EMA codebook) → Decoder → reconstruction loss
2. **Phase 2 — Prior training**: Freeze VQ-VAE, encode all training samples to discrete indices, train an autoregressive prior (Gated PixelCNN for images, WaveNet1DPrior for audio, action-conditioned Transformer for video) on those indices

### Shared component: `VectorQuantizerEMA`

Defined identically in both [Image_Ex/Model.py](Image_Ex/Model.py) and [Audio_Ex/Model.py](Audio_Ex/Model.py). The audio branch adapts 1D sequences to the 2D VQ interface via `unsqueeze(-1)` / `squeeze(-1)`. Key mechanism:
- Codebook weights are buffers updated via EMA (not SGD), with Laplace smoothing to prevent dead codes
- Straight-through gradient estimator: `quantized = z_e + (quantized - z_e).detach()`

### Training loss conventions

| Modality | Reconstruction loss | Total loss |
|----------|-------------------|------------|
| Image    | `MSELoss(x_recon, x)` | `MSE + commit_loss` |
| Audio    | `CrossEntropyLoss(logits, mu_law_target)` | `CE + commit_loss` |
| Video    | `MSELoss(x_recon, x)` | `MSE + commit_loss` |

Image data is normalized (mean/std per dataset), audio is μ-law encoded to 256 classes.

### Checkpoint format

All `.pt` files in `checkpoints/` contain at minimum `model_state_dict` and `config`. The audio checkpoint also stores `speaker_map`. Pre-encoded index caches (`vqvae_indices_dataset.pt`, `vqvae_audio_token_dataset.pt`) accelerate prior training.

### Notebooks are the primary interface

The `.ipynb` files in `Image_Ex/`, `Audio_Ex/`, and `Video_Ex/` are the intended entry points — they import from `Model.py`, `Train.py`, `GetData.py` and orchestrate the full training/generation workflow. The `.py` files are library modules, not standalone scripts.

### Known quirks

- `from turtle import forward` at the top of both `Model.py` files is a stray/unused import — harmless but should not be replicated
- `Image_Ex/GetData.py` imports `soundfile` and `torchaudio` even though image loading doesn't need them — they're only used by `Audio_Ex/GetData.py`
- `src/` is empty; all code was migrated to `Image_Ex/` and `Audio_Ex/`
- The CIFAR notebook filename spells "CIAFAR" — intentional historical typo, not a bug
- Two different `vqvae_labels_dataset.pt` files exist (one for images, one for audio) in different runtime contexts but share the same filename — be careful when moving/copying
- `Video_Ex/` is a work-in-progress: `GetData.py` is implemented (Atari DQN replay via torchrl), but `Model.py` and `Train.py` don't exist yet — notebooks contain interface specs as commented-out code

## Key hyperparameters

```python
# Image VQ-VAE
HIDDEN_CHANNELS = 128, RESIDUAL_CHANNELS = 32, RESIDUAL_LAYERS = 2
NUM_EMBEDDINGS = 512, EMBEDDING_DIM = 64
DECAY = 0.99, COMMIT_COST = 0.25

# Audio VQ-VAE (adds WaveNet-specific)
wn_residual_channels = 128, wn_skip_channels = 256, wn_kernel_size = 2
wn_dilation_cycles = 2, wn_layers_per_cycle = 10

# Both use Adam with lr=2e-4, weight_decay=1e-6
```

## Monitoring codebook health

- **Perplexity** close to `K` (512) = good utilization; near 1 = codebook collapse
- Use `codebook_usage_audio()` in [Audio_Ex/Train.py](Audio_Ex/Train.py) for per-code usage stats
- The image branch tracks perplexity per epoch but doesn't have a dedicated `codebook_usage` function — if needed, port the audio version's pattern
