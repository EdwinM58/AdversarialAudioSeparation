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

# Separate audio (after training)
python separate.py --input song.wav
python separate.py --input song.mp3 --output results/ --checkpoint checkpoints/542621_sup/542621_sup-1001
```

There is no test suite or linter configured. Ruff is available as a dev dependency.

## Environment Setup

The conda environment `adv-audio-sep` is used for development:

```bash
eval "$(/opt/homebrew/bin/brew shellenv)"  # ensure ffmpeg is on PATH
conda activate /Users/edwinmongare/anaconda3/envs/adv-audio-sep
```

## Dependencies

- Python 3.11+
- TensorFlow 2.x (uses `tf.compat.v1` bridge for session-based execution)
- `tf-keras` package required (TF 2.16+ ships Keras 3; legacy Keras 2 is needed for `tf.layers.*`)
- `TF_USE_LEGACY_KERAS=1` environment variable is set at the top of all TF-importing files
- Sacred for experiment configuration
- librosa 0.10+ (keyword-only args for `stft`/`istft`), soundfile for audio I/O
- musdb, stempeg for MUSDB18 dataset loading
- mir_eval for BSS evaluation metrics (SDR, SIR, SAR)
- ffmpeg must be on PATH for mp3 support and stempeg

Install via `pip install -e .` or `pip install -r requirements.txt`.

## Architecture

### Training Pipeline (`Training.py`)

The entry point uses Sacred's `@ex.automain`. On first run, it loads datasets via `getMUSDB18()` (and optionally `getMoisesDB()`), builds a `dataset.pkl` cache, then trains:
1. **Supervised phase**: U-Net separator trained with MSE loss on paired (mixture, accompaniment, voice) data from MUSDB18
2. **Semi-supervised phase**: Same architecture retrained with supervised MSE + adversarial WGAN loss (weighted by `alpha`) + additive mask penalty (weighted by `beta`) using unpaired data (MoisesDB if available)

Both phases use `optimise()` which loops: `train()` one epoch → `test()` on validation → early stop → final `Test.bss_evaluate()` on test set.

### Inference (`separate.py`)

Standalone CLI script for separating audio files using a trained checkpoint. Auto-detects the latest checkpoint from `checkpoints/`. Performs sliding window inference over the input spectrogram and reconstructs vocals and accompaniment wav files.

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

Two modern dataset loaders:
- **`getMUSDB18(musdb_path, is_wav=False)`**: Uses the `musdb` Python package to read MUSDB18 `.stem.mp4` files. Exports stems (mix, vocals, accompaniment) to wav on first run. Returns `[train_list, test_list]` of Sample tuples. 100 training tracks (supervised), 50 test tracks split into 25 validation + 25 test.
- **`getMoisesDB(moisesdb_path)`**: Scans MoisesDB folder structure, classifies stems as vocal/non-vocal by keyword matching, combines non-vocal stems into accompaniment, exports to wav. Used for unsupervised training data.

Legacy XML-based loaders (`getDSDFilelist`, `getCCMixter`, `getIKala`, `getMedleyDB`) remain in the file but are unused.

Datasets should be placed in the `datasets/` directory (gitignored). MUSDB18 can be symlinked from another location.

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
- All files that import TensorFlow must set `os.environ['TF_USE_LEGACY_KERAS'] = '1'` before importing TF
- TF1 APIs use `tf.compat.v1 as tf1`; math ops use `tf.math.log1p` / `tf.math.expm1`
- librosa calls use keyword arguments: `librosa.stft(audio, n_fft=..., hop_length=...)`

## Related: HarmonySplit Web App

A companion web application lives at `../harmony_split/`. It provides a React frontend (Vite + Tailwind) and Django backend that wraps this model for browser-based audio separation. The backend's `model/inference.py` imports from this repo via `sys.path`.
