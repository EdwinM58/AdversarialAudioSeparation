#!/usr/bin/env python3
"""
Separate a song into vocals and accompaniment using a trained model.

Usage:
    python separate.py --input song.mp3 --checkpoint checkpoints/123_sup/123_sup-1001 --output_dir results/

    # Use Griffin-Lim phase refinement (slower but can improve quality):
    python separate.py --input song.mp3 --checkpoint checkpoints/123_sup/123_sup-1001 --phase_iterations 10
"""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import argparse
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
tf1.disable_v2_behavior()
import librosa
import soundfile as sf

import Models.Unet
import Utils
from Input import Input


def find_latest_checkpoint(checkpoints_dir="checkpoints"):
    """Find the most recent checkpoint in the checkpoints directory."""
    if not os.path.isdir(checkpoints_dir):
        return None

    latest = None
    latest_step = -1

    for folder in os.listdir(checkpoints_dir):
        folder_path = os.path.join(checkpoints_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        ckpt = tf1.train.latest_checkpoint(folder_path)
        if ckpt:
            # Extract step number from checkpoint name
            try:
                step = int(ckpt.split("-")[-1])
                if step > latest_step:
                    latest_step = step
                    latest = ckpt
            except ValueError:
                if latest is None:
                    latest = ckpt

    return latest


def separate(input_path, checkpoint_path, output_dir, model_config, phase_iterations=0):
    """
    Separate an audio file into vocals and accompaniment.

    :param input_path: Path to the input audio file (wav, mp3, etc.)
    :param checkpoint_path: Path to the trained model checkpoint
    :param output_dir: Directory to save the separated sources
    :param model_config: Model configuration dict
    :param phase_iterations: Number of Griffin-Lim iterations (0 = use mixture phase directly)
    """
    # Determine input and output shapes
    freq_bins = model_config["num_fft"] // 2 + 1
    disc_input_shape = [1, freq_bins - 1, model_config["num_frames"], 1]

    separator_class = Models.Unet.Unet(model_config["num_layers"])
    sep_input_shape, sep_output_shape = separator_class.getUnetPadding(np.array(disc_input_shape))
    separator_func = separator_class.get_output

    # Build the separator graph
    mix_context = tf1.placeholder(tf.float32, shape=sep_input_shape, name="input")
    mix_context_norm = Input.norm(mix_context)

    separator_acc_norm, separator_voice_norm = separator_func(mix_context_norm, reuse=False)
    separator_acc = Input.denorm(separator_acc_norm)
    separator_voice = Input.denorm(separator_voice_norm)

    # Start session and load model
    sess = tf1.Session()
    sess.run(tf1.global_variables_initializer())

    restorer = tf1.train.Saver(None, write_version=tf1.train.SaverDef.V2)
    restorer.restore(sess, checkpoint_path)
    print(f"Model restored from: {checkpoint_path}")

    # Load and preprocess audio
    print(f"Loading: {input_path}")
    mix_audio, orig_sr = librosa.load(input_path, sr=model_config["expected_sr"], mono=True)
    mix_length = len(mix_audio)
    duration_sec = mix_length / model_config["expected_sr"]
    print(f"Audio: {duration_sec:.1f}s at {model_config['expected_sr']}Hz ({mix_length} samples)")

    # Pad for STFT/ISTFT roundtrip
    mix_audio_pad = librosa.util.fix_length(mix_audio, size=mix_length + model_config["num_fft"] // 2)

    # Compute STFT
    mix_mag, mix_ph = Input.audioFileToSpectrogram(mix_audio_pad, model_config["num_fft"], model_config["num_hop"])
    source_time_frames = mix_mag.shape[1]

    # Preallocate output spectrograms
    acc_pred_mag = np.zeros(mix_mag.shape, np.float32)
    voice_pred_mag = np.zeros(mix_mag.shape, np.float32)

    input_time_frames = sep_input_shape[2]
    output_time_frames = sep_output_shape[2]

    # Pad mixture spectrogram along time for U-Net context
    pad_time_frames = (input_time_frames - output_time_frames) // 2
    mix_mag_padded = np.pad(mix_mag, [(0, 0), (pad_time_frames, pad_time_frames)], mode="constant", constant_values=0.0)

    # Sliding window inference
    num_windows = (source_time_frames + output_time_frames - 1) // output_time_frames
    print(f"Running inference ({num_windows} windows)...")

    for source_pos in range(0, source_time_frames, output_time_frames):
        if source_pos + output_time_frames > source_time_frames:
            source_pos = source_time_frames - output_time_frames

        # Extract input patch
        mix_mag_part = mix_mag_padded[:, source_pos:source_pos + input_time_frames]
        mix_mag_part = Utils.pad_freqs(mix_mag_part, sep_input_shape[1:3])
        mix_mag_part = mix_mag_part[np.newaxis, :, :, np.newaxis]

        # Run through network
        acc_part, voice_part = sess.run(
            [separator_acc, separator_voice],
            feed_dict={mix_context: mix_mag_part}
        )

        # Store predictions (drop the padded frequency bin)
        acc_pred_mag[:, source_pos:source_pos + output_time_frames] = acc_part[0, :-1, :, 0]
        voice_pred_mag[:, source_pos:source_pos + output_time_frames] = voice_part[0, :-1, :, 0]

    # Convert spectrograms back to audio
    print("Reconstructing audio...")
    acc_audio = Input.spectrogramToAudioFile(
        acc_pred_mag, model_config["num_fft"], model_config["num_hop"],
        phase=mix_ph, length=mix_length, phaseIterations=phase_iterations
    )
    voice_audio = Input.spectrogramToAudioFile(
        voice_pred_mag, model_config["num_fft"], model_config["num_hop"],
        phase=mix_ph, length=mix_length, phaseIterations=phase_iterations
    )

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(input_path))[0]

    vocals_path = os.path.join(output_dir, f"{basename}_vocals.wav")
    acc_path = os.path.join(output_dir, f"{basename}_accompaniment.wav")
    mix_out_path = os.path.join(output_dir, f"{basename}_original.wav")

    sf.write(vocals_path, voice_audio, model_config["expected_sr"])
    sf.write(acc_path, acc_audio, model_config["expected_sr"])
    sf.write(mix_out_path, mix_audio, model_config["expected_sr"])

    print(f"\nSaved:")
    print(f"  Vocals:        {vocals_path}")
    print(f"  Accompaniment: {acc_path}")
    print(f"  Original (8kHz): {mix_out_path}")

    sess.close()
    tf1.reset_default_graph()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Separate vocals from accompaniment")
    parser.add_argument("--input", "-i", required=True, help="Path to input audio file (wav, mp3, etc.)")
    parser.add_argument("--checkpoint", "-c", default=None, help="Path to model checkpoint (auto-detects latest if omitted)")
    parser.add_argument("--output_dir", "-o", default="separated", help="Output directory (default: separated/)")
    parser.add_argument("--phase_iterations", "-p", type=int, default=0,
                        help="Griffin-Lim phase iterations (0=use mixture phase, 10=refine phase)")
    parser.add_argument("--num_layers", type=int, default=4, help="U-Net layers (must match trained model)")
    args = parser.parse_args()

    # Model config matching Training.py defaults
    model_config = {
        "num_fft": 512,
        "num_hop": 256,
        "expected_sr": 8192,
        "mono_downmix": True,
        "num_frames": 64,
        "num_layers": args.num_layers,
        "batch_size": 1,
    }

    # Auto-detect checkpoint if not specified
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = find_latest_checkpoint()
        if checkpoint is None:
            print("Error: No checkpoint found. Train a model first or specify --checkpoint.")
            exit(1)
        print(f"Auto-detected checkpoint: {checkpoint}")

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        exit(1)

    separate(args.input, checkpoint, args.output_dir, model_config, args.phase_iterations)
