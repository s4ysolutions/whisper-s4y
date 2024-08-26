import os
import logging
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
import tensorflow_io as tfio
import whisper_s4y.whisper.huggingface as hf

from matplotlib.axes import Axes

test_log = logging.getLogger(__name__)
_log_handler = logging.StreamHandler(sys.stdout)
_log_handler.setFormatter(logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s"))
test_log.addHandler(_log_handler)
test_log.setLevel(logging.DEBUG)


def _plot_audio(audio: tf.TensorSpec(shape=[480000], dtype=tf.float32), ax: Axes, title: str = 'Audio'):
    time = [float(n) / 16000 for n in range(len(audio))]
    ax.plot(time, audio)
    ax.set_title(title)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Amplitude')


def _plot_logmel(logmel: tf.TensorSpec(shape=[80, 3000], dtype=tf.float32), fig, ax: Axes, title: str = 'LogMel'):
    # cax = ax.imshow(tf.transpose(logmel), aspect='auto', origin='lower', cmap='viridis')
    cax = ax.imshow(logmel, aspect='auto', origin='lower', cmap='viridis')
    fig.colorbar(cax, ax=ax, label='Magnitude')
    ax.set_title(title)
    ax.set_ylabel('Frequency bin')
    ax.set_xlabel('Time frame')


def _plot_output(logmel: tf.TensorSpec(shape=[1500, 384], dtype=tf.float32), fig, ax: Axes,
                 title: str = 'Encoded output'):
    # cax = ax.imshow(tf.transpose(logmel), aspect='auto', origin='lower', cmap='viridis')
    cax = ax.imshow(logmel, aspect='auto', origin='lower', cmap='viridis')
    fig.colorbar(cax, ax=ax, label='Value')
    ax.set_title(title)
    ax.set_ylabel('Frames')
    ax.set_xlabel('Embeddings')


def plot_input_features(features_transformers: tf.TensorSpec(shape=[1, 80, 30000], dtype=tf.float32),
                        features_under_test: tf.TensorSpec(shape=[1, 80, 3000], dtype=tf.float32), title):
    fig1, axs1 = plt.subplots(2, 1, figsize=(12, 12))  # Create a figure containing a single Axes.
    fig1.canvas.manager.set_window_title(f"Test: {title}")
    # fig1.suptitle(f"Audio: {title}")
    _plot_logmel(features_under_test[0], fig1, axs1[0], f"LogMel (under test)")
    _plot_logmel(features_transformers[0], fig1, axs1[1], f"LogMel (transformers)")
    plt.show()


def plot_diff(tf1: tf.TensorSpec(shape=None, dtype=tf.float32), tf2: tf.TensorSpec(shape=None, dtype=tf.float32),
              title):
    tf0 = tf1 - tf2
    fig1, ax = plt.subplots(1, 1, figsize=(12, 12))  # Create a figure containing a single Axes.
    fig1.canvas.manager.set_window_title(f"Test: {title}")
    cax = ax.imshow(tf0, aspect='auto', origin='lower', cmap='viridis')
    fig1.colorbar(cax, ax=ax, label='Magnitude')

#plot_diff(tf.constant([1, 2, 3, 4, 5], dtype=tf.float32), tf.constant([5, 4, 3, 2, 1], dtype=tf.float32), "test")

def plot_encoded_output(output_transformers: tf.TensorSpec(shape=[1, 1500, 384], dtype=tf.float32),
                        output_under_test: tf.TensorSpec(shape=[1, 1500, 384], dtype=tf.float32), title):
    fig1, axs1 = plt.subplots(2, 1, figsize=(12, 12))  # Create a figure containing a single Axes.
    fig1.canvas.manager.set_window_title(f"Test: {title}")
    # fig1.suptitle(f"Audio: {title}")
    _plot_output(output_under_test[0], fig1, axs1[0], f"Encoded output (under test)")
    _plot_output(output_transformers[0], fig1, axs1[1], f"Encoded output (transformers)")
    plt.show()


_cwd = os.path.dirname(os.path.abspath(__file__))
_tests_dir = _cwd
_data_dir = os.path.join(_tests_dir, "data")

test_model_id = 'openai/whisper-tiny'


def test_data_path(file_name: str) -> str:
    return os.path.join(_data_dir, file_name)


def normalized_audio_from_wav(wav_file_path) -> tf.TensorSpec(shape=[480000], dtype=tf.float32):
    if not os.path.isabs(wav_file_path):
        wav_file_path = os.path.join(_data_dir, wav_file_path)
    waveform, sample_rate = tf.audio.decode_wav(tf.io.read_file(wav_file_path))
    if sample_rate != 16000:
        waveform = tfio.audio.resample(waveform, rate_in=sample_rate, rate_out=16000)
    audio = waveform[:, 0]
    size = audio.shape[0]
    if size < 480000:
        audio = tf.concat([audio, tf.zeros(480000 - size)], 0)
    elif size > 480000:
        audio = audio[:480000]
    return audio


def transformers_input_features(audio: tf.TensorSpec(shape=[480000], dtype=tf.float32),
                                huggingface_model_id: str = test_model_id) -> tf.TensorSpec(shape=[1, 80, 3000],
                                                                                            dtype=tf.float32):
    feature_extractor = hf.feature_extractor(huggingface_model_id)
    return feature_extractor(audio, sampling_rate=16000, return_tensors="tf")["input_features"]


def transformers_encoded_output(input_features: tf.TensorSpec(shape=[1, 80, 3000], dtype=tf.float32),
                                huggingface_model_id: str = test_model_id) -> tf.TensorSpec(
    shape=[1, 5000, 384],
    dtype=tf.float32):
    encoder = hf.encoder(huggingface_model_id)
    output = encoder(input_features)
    return output.last_hidden_state
