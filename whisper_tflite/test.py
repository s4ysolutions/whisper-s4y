import argparse
import os.path
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio
from matplotlib.axes import Axes
from typing import Optional
from transformers import WhisperProcessor

_cwd = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_cwd)


def normalized_audio_from_wav(wav_file_path) -> tf.TensorSpec(shape=[480000], dtype=tf.float32):
    waveform, sample_rate = tf.audio.decode_wav(tf.io.read_file(wav_file_path))
    if sample_rate != 16000:
        print(f"sample rate is {sample_rate}, resampling...")
        waveform = tfio.audio.resample(waveform, rate_in=sample_rate, rate_out=16000)
        print("ok")
    audio = waveform[:, 0]
    size = audio.shape[0]
    if size < 480000:
        audio = tf.concat([audio, tf.zeros(480000 - size)], 0)
    elif size > 480000:
        audio = audio[:480000]
    return audio


def tflite_interpreter(tflite_model_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    runner = interpreter.get_signature_runner()
    return runner, interpreter


def extract_features(audio, extractor: Optional[tf.function] = None):
    if extractor is None:
        extractor, _ = features_extractor()
    return extractor(normalized_audio=audio)


def features_extractor() -> tuple[tf.function, tf.lite.Interpreter]:
    return tflite_interpreter(os.path.join(_root, "artefacts/features-extractor.tflite"))


def features_extractor_transformers() -> tuple[tf.function, tf.lite.Interpreter]:
    return WhisperProcessor.from_pretrained('openai/whisper-tiny')


def extract_features_transformers(audio: tf.TensorSpec(shape=[480000], dtype=tf.float32),
                                  extractor: Optional[tf.function] = None):
    if extractor is None:
        extractor = features_extractor_transformers()
    return extractor(audio, sampling_rate=16000, return_tensors="tf")


def audio_en_1():
    return normalized_audio_from_wav(os.path.join(_root, 'test_data', 'en', 'OSR_us_000_0030_16k.wav')), 'en 1'


def plot_audio(audio: tf.TensorSpec(shape=[480000], dtype=tf.float32), ax: Axes, title: str = 'Audio'):
    time = [float(n) / 16000 for n in range(len(audio))]
    ax.plot(time, audio)
    ax.set_title(title)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Amplitude')


def plot_logmel(logmel: tf.TensorSpec(shape=[80, 3000], dtype=tf.float32), ax: Axes, title: str = 'LogMel'):
    cax = ax.imshow(logmel.T, aspect='auto', origin='lower', cmap='viridis')
    fig.colorbar(cax, ax=ax, label='Magnitude')
    ax.set_title(title)
    ax.set_xlabel('Frequency bin')
    ax.set_ylabel('Time frame')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test package clie")

    parser.add_argument("--skip_plot", type=str, help="Skip the generator model", default=False)
    args = parser.parse_args()

    audio1, title = audio_en_1()
    features1 = extract_features(audio1)['logmel']
    features1t = extract_features_transformers(audio1)['input_features'][-1]

    if not args.skip_plot:
        fig, axs = plt.subplots(1, 2,  figsize=(12, 12))  # Create a figure containing a single Axes.
        plot_logmel(features1, axs[0], f"LogMel (tflite)")
        plot_logmel(features1, axs[1], f"LogMel (transformers)")
        plt.show()
