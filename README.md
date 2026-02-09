# AdversarialAudioSeparation

Code accompanying the paper "Semi-supervised adversarial audio source separation applied to singing voice extraction" available on arXiv here:

https://arxiv.org/abs/1711.00048

## The idea

Improve existing supervised audio source separation models, which are commonly neural networks, with extra unlabelled mixture recordings as well as unlabelled solo recordings of the sources we want to separate. The network is trained in a normal supervised fashion to minimise its prediction error on fully annotated data (samples with mixture and sources paired up), and at the same time to output source estimates for the extra mixture recordings that are indistinguishable from the solo source recordings.

To achieve this, we use adversarial training: One discriminator network is trained per source to identify whether a source excerpt comes from the real solo source recordings or from the separator when evaluated on the extra mixtures.

This can prevent overfitting to the often small annotated dataset and makes use of the much more easily available unlabelled data.

<img src="./system_diagram.png" width="700">

## Setup

### Requirements

- Python 3.11+
- TensorFlow 2.x (GPU version recommended due to long running times)
- tf-keras (required for TF 2.16+ which ships Keras 3 by default)
- ffmpeg must be installed and on your PATH for mp3 support and MUSDB18 stem reading

Install dependencies:

```bash
pip install -r requirements.txt
```

Or install as a package in development mode:

```bash
pip install -e .
```

### Dataset preparation

The codebase uses [MUSDB18](https://sigsep.github.io/datasets/musdb.html) for supervised training and optionally [MoisesDB](https://github.com/moises-ai/moises-db) for semi-supervised training.

When the code is run for the first time, it exports dataset stems to wav and creates a `dataset.pkl` cache, so that subsequent starts are much faster.

#### MUSDB18 (required)

1. Download MUSDB18 (the compressed `.stem.mp4` version or the HQ uncompressed wav version)
2. Place or symlink it at `datasets/MUSDB18`:

```bash
mkdir -p datasets
ln -s /path/to/musdb18 datasets/MUSDB18
```

The `musdb` Python package handles reading the `.stem.mp4` files. On first run, stems are exported to wav files alongside the originals. This provides 100 supervised training tracks and 50 test tracks (split 25 validation / 25 test).

To use the HQ wav version instead, change `is_wav=False` to `is_wav=True` in `Training.py`.

#### MoisesDB (optional, for semi-supervised training)

1. Request access at the [MoisesDB repository](https://github.com/moises-ai/moises-db)
2. Place or symlink it at `datasets/MoisesDB`

MoisesDB provides ~240 additional tracks used as unpaired data for the adversarial semi-supervised phase. If not available, training runs in supervised-only mode.

#### Option: Use your own data

To use custom datasets, replace the data loading code in `Training.py` with a function that returns a dictionary mapping:

```
"train_sup" : sample_list          # list of (mix, acc, voice) Sample tuples
"train_unsup" : [mix_list, acc_list, voice_list]  # unpaired lists
"train_valid" : sample_list
"train_test" : sample_list
```

See `Sample.py` for the Sample class constructor.

### Configuration and hyperparameters

You can configure settings and hyperparameters by modifying the `model_config` dictionary defined in the beginning of `Training.py` or using the commandline features of sacred by setting certain values when calling the script via commandline (see Sacred documentation).

Note that alpha and beta (hyperparameters from the paper) as loss weighting parameters are relatively important for good performance, tweaking these might be necessary. These are also editable in the `model_config` dictionary.

## Training

The code is run by executing

```bash
python Training.py
```

It will train the same separator network first in a purely supervised way, and then using our semi-supervised adversarial approach. Each time, validation performance is measured regularly and early stopping is used, before the final test set performance is evaluated. For the semi-supervised approach, the additional data from `dataset["train_unsup"]` is used to improve performance.

Finally, BSS evaluation metrics are computed on the test dataset (SDR, SIR, SAR) - this saves the results in a pickled file along with the name of the dataset, so if you aim to use different datasets, the function needs to be extended slightly.

Logs are written continuously to the logs subfolder, so training can be supervised with Tensorboard. Checkpoint files of the model are created whenever validation performance is tested.

## Inference

After training, separate audio files using the standalone inference script:

```bash
# Auto-detect latest checkpoint
python separate.py --input song.wav

# Specify checkpoint and output directory
python separate.py --input song.mp3 --output results/ --checkpoint checkpoints/542621_sup/542621_sup-1001
```

This outputs `<filename>_vocals.wav`, `<filename>_accompaniment.wav`, and a copy of the original in the output directory (default: `separated/`).

## Web Application

A companion web app for browser-based audio separation will be available at [harmony_split](../harmony_split/). It provides a React frontend with drag-and-drop upload and a Django backend that wraps the trained model for HTTP-based inference.
