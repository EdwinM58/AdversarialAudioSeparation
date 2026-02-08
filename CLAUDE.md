# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Semi-supervised adversarial audio source separation for singing voice extraction. Implements the paper "Semi-supervised adversarial audio source separation applied to singing voice extraction" (arXiv:1711.00048). The system separates audio mixtures into vocal and accompaniment tracks using a U-Net separator trained with WGAN-based adversarial loss on unlabelled data.

## Commands

```bash
# Train (runs supervised then semi-supervised training with early stopping, then BSS evaluation)
python Training.py

# Override hyperparameters via Sacred CLI
python Training.py with "model_config={'alpha': 0.01, 'batch_size': 32}"

# Monitor training
tensorboard --logdir=logs
```

There is no test suite or linter configured. Ruff is available as a dev dependency.

## Dependencies

- Python 3.11+
- TensorFlow 2.x (uses `tf.compat.v1` bridge for session-based execution)
- Sacred for experiment configuration
- librosa, soundfile for audio I/O
- mir_eval for BSS evaluation metrics (SDR, SIR, SAR)
- ffmpeg must be on PATH for mp3 support

Install via `pip install -e .` or `pip install -r requirements.txt`.

## Architecture

### Training Pipeline (`Training.py`)

The entry point uses Sacred's `@ex.automain`. On first run, it reads dataset XML files, builds a `dataset.pkl` cache, then trains:
1. **Supervised phase**: U-Net separator trained with MSE loss on paired (mixture, accompaniment, voice) data
2. **Semi-supervised phase**: Same architecture retrained with supervised MSE + adversarial WGAN loss (weighted by `alpha`) + additive mask penalty (weighted by `beta`) using unpaired data

Both phases use `optimise()` which loops: `train()` one epoch → `test()` on validation → early stop → final `Test.bss_evaluate()` on test set.

### Separator (`Models/Unet.py`)

U-Net with valid convolutions operating on magnitude spectrograms `[batch, freqs, time, 1]`. `getUnetPadding()` computes the required input shape for a desired output shape (input is larger due to valid convolutions). The output is two branches: accompaniment and voice log-magnitude estimates.

### Discriminators (`Models/WGAN_Critic.py`)

WGAN with gradient penalty. Two independent critics (one per source: `acc_disc`, `voice_disc`) using a DCGAN-style architecture. `create_critic()` builds the full WGAN loss including gradient penalty term weighted by `lam`.

### Data Loading (`Input/`)

Two-tier system:
- **`batchgenerators.py`**: `BatchGen_Paired` (supervised, matched mix/acc/voice triples) and `BatchGen_Single` (unsupervised, individual streams). These draw random patches from a cache.
- **`multistreamcache.py` / `multistreamworkers.py`**: Background workers read audio files, compute spectrograms, and fill a shared cache. Cache entries are replaced at `min_replacement_rate` per batch.
- **`Input.py`**: Audio I/O (wav via SoundFile, mp3 via ffmpeg subprocess), STFT/ISTFT, Griffin-Lim phase reconstruction, log-normalization (`norm`/`denorm` = `log1p`/`expm1`).

### Dataset Preparation (`Datasets.py`)

Reads XML metadata files (DSD100.xml, MedleyDB.xml, CCMixter.xml, iKala.xml) pointing to dataset root folders. Creates `Sample` objects with audio path and metadata. For datasets without separate accompaniment tracks, generates them by signal subtraction (`subtract_audio`).

Dataset partitioning: DSD100 training = supervised, DSD100 test split into validation (25) and test (25), MedleyDB/CCMixter/iKala each split into thirds for unsupervised/validation/test.

### Key Hyperparameters (in `model_config`)

- `alpha`, `beta`: Loss weights for adversarial and mask penalty terms — critical for performance
- `num_layers`: U-Net depth (default 4)
- `num_fft`/`num_hop`: STFT parameters (512/256)
- `expected_sr`: All audio downsampled to 8192 Hz
- `num_disc`: Discriminator updates per separator update (default 5)
- `epoch_it`: Steps per training epoch (default 1000)

## Key Conventions

- Spectrograms are 4D tensors: `[batch_size, frequencies, time_frames, 1]`
- Frequency axis is `num_fft/2` (256 bins, dropping the last to make even)
- Magnitudes are log-normalized (`log1p`) before network processing
- `Utils.pad_freqs` / `Utils.crop_and_concat` handle shape mismatches between U-Net layers
- Variable scoping: separator under `"separator"`, discriminators under `"acc_disc"` / `"voice_disc"`
