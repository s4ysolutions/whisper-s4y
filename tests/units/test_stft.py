import numpy as np
import tensorflow as tf
import torch

from whisper_s4y.features_extractor import FRAME_LENGTH, FRAME_STEP, FFT_LENGTH, window_function as tf_window_function
from transformers.audio_utils import window_function as torch_window_function


def test_tf_window_fn():
    window = tf_window_function(FRAME_LENGTH)
    assert window.shape == (FRAME_LENGTH,)


def test_pt_window_fn(waveform_ar):
    window = torch.from_numpy(torch_window_function(FRAME_LENGTH, "hann"))
    assert window.shape == (FRAME_LENGTH,)


def test_tf_stft_doesnt_obey_window_length(waveform_ar):
    # Arrange
    normalized_audio = waveform_ar
    window = tf_window_function(FRAME_LENGTH)
    window_length = window.shape[0]
    # Act
    stft_tensor = tf.signal.stft(normalized_audio, frame_length=FRAME_LENGTH, frame_step=FRAME_STEP, pad_end=True,
                                 fft_length=FFT_LENGTH, window_fn=tf_window_function)
    # Assert
    assert stft_tensor.shape[0] == 3000
    assert stft_tensor.shape[1] != window_length // 2 + 1
    assert stft_tensor.shape[1] == FFT_LENGTH // 2 + 1


def test_pt_stft_obey_window_length(waveform_ar):
    # Arrange
    normalized_audio = torch.from_numpy(waveform_ar.numpy())
    window = torch.from_numpy(torch_window_function(FRAME_LENGTH, "hann"))
    window_length = window.shape[0]
    # Act
    stft_tensor = torch.stft(normalized_audio, n_fft=FRAME_LENGTH, hop_length=FRAME_STEP, window=window,
                             return_complex=True)
    # Assert
    assert stft_tensor.shape[1] == 3001
    assert stft_tensor.shape[0] == window_length // 2 + 1
    assert stft_tensor.shape[0] != FFT_LENGTH // 2 + 1


def test_tf_stft_returns_zeroes_outside_window(waveform_ar):
    # Arrange
    normalized_audio = waveform_ar
    window_length = tf_window_function(FRAME_LENGTH).shape[0]
    n_bins_in_window = window_length // 2 + 1
    # Act
    stft_tensor = tf.signal.stft(normalized_audio, frame_length=FRAME_LENGTH, frame_step=FRAME_STEP, pad_end=True,
                                 fft_length=FFT_LENGTH,
                                 window_fn=tf_window_function)
    magnitudes = tf.square(tf.abs(stft_tensor))
    # Assert all outside of windows is zero and can be removed
    # Act
    # Assert all on the left are zeroes
    #tf.where(tf.greater(magnitudes[:, -56:], 1e-2))
    bins_outside_window = magnitudes[:, n_bins_in_window:]

    # there are few non-zero values beyond the window
    assert tf.where(tf.greater(magnitudes[:, n_bins_in_window:], 1e-2)).shape[0] == 0
    assert tf.where(tf.greater(magnitudes[:, n_bins_in_window:], 1e-3)).shape[0] < 60
    assert tf.reduce_max(tf.where(tf.greater(magnitudes[:, 200:], 1e-3))[:,1]).numpy() < 5
    assert tf.where(tf.greater(magnitudes[:, n_bins_in_window:], 1e-4)).shape[0] < 500
    assert tf.reduce_max(tf.where(tf.greater(magnitudes[:, 200:], 1e-4))[:,1]).numpy() < 12
    assert tf.where(tf.greater(magnitudes[:, n_bins_in_window:], 1e-5)).shape[0] < 2500
    assert tf.reduce_max(tf.where(tf.greater(magnitudes[:, 200:], 1e-5))[:,1]).numpy() < 20
    assert tf.where(tf.greater(magnitudes[:, n_bins_in_window:], 1e-6)).shape[0] < 6000
    assert tf.reduce_max(tf.where(tf.greater(magnitudes[:, 200:], 1e-6))[:,1]).numpy() < 25
