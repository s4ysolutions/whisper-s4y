import logging
import numpy as np
import os
import tempfile
import tensorflow as tf

log = logging.getLogger("whisper2tfilte")


def hertz_to_mel(freq: float, mel_scale: str = "htk"):
    """
    Convert frequency from hertz to mels.

    Args:
        freq (`float` or `np.ndarray`):
            The frequency, or multiple frequencies, in hertz (Hz).
        mel_scale (`str`, *optional*, defaults to `"htk"`):
            The mel frequency scale to use, `"htk"`, `"kaldi"` or `"slaney"`.

    Returns:
        `float` or `np.ndarray`: The frequencies on the mel scale.
    """

    if mel_scale not in ["slaney", "htk", "kaldi"]:
        raise ValueError('mel_scale should be one of "htk", "slaney" or "kaldi".')

    if mel_scale == "htk":
        return 2595.0 * np.log10(1.0 + (freq / 700.0))
    elif mel_scale == "kaldi":
        return 1127.0 * np.log(1.0 + (freq / 700.0))

    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = 27.0 / np.log(6.4)
    mels = 3.0 * freq / 200.0

    if isinstance(freq, np.ndarray):
        log_region = freq >= min_log_hertz
        mels[log_region] = min_log_mel + np.log(freq[log_region] / min_log_hertz) * logstep
    elif freq >= min_log_hertz:
        mels = min_log_mel + np.log(freq / min_log_hertz) * logstep
    return mels


def mel_to_hertz(mels: float, mel_scale: str = "htk"):
    """
    Convert frequency from mels to hertz.

    Args:
        mels (`float` or `np.ndarray`):
            The frequency, or multiple frequencies, in mels.
        mel_scale (`str`, *optional*, `"htk"`):
            The mel frequency scale to use, `"htk"`, `"kaldi"` or `"slaney"`.

    Returns:
        `float` or `np.ndarray`: The frequencies in hertz.
    """

    if mel_scale not in ["slaney", "htk", "kaldi"]:
        raise ValueError('mel_scale should be one of "htk", "slaney" or "kaldi".')

    if mel_scale == "htk":
        return 700.0 * (np.power(10, mels / 2595.0) - 1.0)
    elif mel_scale == "kaldi":
        return 700.0 * (np.exp(mels / 1127.0) - 1.0)

    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = np.log(6.4) / 27.0
    freq = 200.0 * mels / 3.0

    if isinstance(mels, np.ndarray):
        log_region = mels >= min_log_mel
        freq[log_region] = min_log_hertz * np.exp(logstep * (mels[log_region] - min_log_mel))
    elif mels >= min_log_mel:
        freq = min_log_hertz * np.exp(logstep * (mels - min_log_mel))

    return freq


def _create_triangular_filter_bank(fft_freqs: np.ndarray, filter_freqs: np.ndarray) -> np.ndarray:
    """
    Creates a triangular filter bank.

    Adapted from *torchaudio* and *librosa*.

    Args:
        fft_freqs (`np.ndarray` of shape `(num_frequency_bins,)`):
            Discrete frequencies of the FFT bins in Hz.
        filter_freqs (`np.ndarray` of shape `(num_mel_filters,)`):
            Center frequencies of the triangular filters to create, in Hz.

    Returns:
        `np.ndarray` of shape `(num_frequency_bins, num_mel_filters)`
    """
    filter_diff = np.diff(filter_freqs)
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    return np.maximum(np.zeros(1), np.minimum(down_slopes, up_slopes))


def mel_filter_bank(
        num_frequency_bins: int,
        num_mel_filters: int,
        min_frequency: float,
        max_frequency: float,
        sampling_rate: int,
        norm=None,
        mel_scale: str = "htk",
        triangularize_in_mel_space: bool = False,
) -> np.ndarray:
    """
    Creates a frequency bin conversion matrix used to obtain a mel spectrogram. This is called a *mel filter bank*, and
    various implementation exist, which differ in the number of filters, the shape of the filters, the way the filters
    are spaced, the bandwidth of the filters, and the manner in which the spectrum is warped. The goal of these
    features is to approximate the non-linear human perception of the variation in pitch with respect to the frequency.

    Different banks of mel filters were introduced in the literature. The following variations are supported:

    - MFCC FB-20: introduced in 1980 by Davis and Mermelstein, it assumes a sampling frequency of 10 kHz and a speech
      bandwidth of `[0, 4600]` Hz.
    - MFCC FB-24 HTK: from the Cambridge HMM Toolkit (HTK) (1995) uses a filter bank of 24 filters for a speech
      bandwidth of `[0, 8000]` Hz. This assumes sampling rate ≥ 16 kHz.
    - MFCC FB-40: from the Auditory Toolbox for MATLAB written by Slaney in 1998, assumes a sampling rate of 16 kHz and
      speech bandwidth of `[133, 6854]` Hz. This version also includes area normalization.
    - HFCC-E FB-29 (Human Factor Cepstral Coefficients) of Skowronski and Harris (2004), assumes a sampling rate of
      12.5 kHz and speech bandwidth of `[0, 6250]` Hz.

    This code is adapted from *torchaudio* and *librosa*. Note that the default parameters of torchaudio's
    `melscale_fbanks` implement the `"htk"` filters while librosa uses the `"slaney"` implementation.

    Args:
        num_frequency_bins (`int`):
            Number of frequencies used to compute the spectrogram (should be the same as in `stft`).
        num_mel_filters (`int`):
            Number of mel filters to generate.
        min_frequency (`float`):
            Lowest frequency of interest in Hz.
        max_frequency (`float`):
            Highest frequency of interest in Hz. This should not exceed `sampling_rate / 2`.
        sampling_rate (`int`):
            Sample rate of the audio waveform.
        norm (`str`, *optional*):
            If `"slaney"`, divide the triangular mel weights by the width of the mel band (area normalization).
        mel_scale (`str`, *optional*, defaults to `"htk"`):
            The mel frequency scale to use, `"htk"`, `"kaldi"` or `"slaney"`.
        triangularize_in_mel_space (`bool`, *optional*, defaults to `False`):
            If this option is enabled, the triangular filter is applied in mel space rather than frequency space. This
            should be set to `true` in order to get the same results as `torchaudio` when computing mel filters.

    Returns:
        `np.ndarray` of shape (`num_frequency_bins`, `num_mel_filters`): Triangular filter bank matrix. This is a
        projection matrix to go from a spectrogram to a mel spectrogram.
    """
    if norm is not None and norm != "slaney":
        raise ValueError('norm must be one of None or "slaney"')

    # center points of the triangular mel filters
    mel_min = hertz_to_mel(min_frequency, mel_scale=mel_scale)
    mel_max = hertz_to_mel(max_frequency, mel_scale=mel_scale)
    mel_freqs = np.linspace(mel_min, mel_max, num_mel_filters + 2)
    filter_freqs = mel_to_hertz(mel_freqs, mel_scale=mel_scale)

    if triangularize_in_mel_space:
        # frequencies of FFT bins in Hz, but filters triangularized in mel space
        fft_bin_width = sampling_rate / (num_frequency_bins * 2)
        fft_freqs = hertz_to_mel(fft_bin_width * np.arange(num_frequency_bins), mel_scale=mel_scale)
        filter_freqs = mel_freqs
    else:
        # frequencies of FFT bins in Hz
        fft_freqs = np.linspace(0, sampling_rate // 2, num_frequency_bins)

    mel_filters = _create_triangular_filter_bank(fft_freqs, filter_freqs)

    if norm is not None and norm == "slaney":
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (filter_freqs[2: num_mel_filters + 2] - filter_freqs[:num_mel_filters])
        mel_filters *= np.expand_dims(enorm, 0)

    if (mel_filters.max(axis=0) == 0.0).any():
        log.warning(
            "At least one mel filter has all zero values. "
            f"The value for `num_mel_filters` ({num_mel_filters}) may be set too high. "
            f"Or, the value for `num_frequency_bins` ({num_frequency_bins}) may be set too low."
        )

    return mel_filters


# These are the constants among the model calls and can be embedded in the graph
frame_length = 400
frame_step = 160
num_mel_bins = 80
# fft_length = 400  # transformers
fft_length = 512  # 2^9 > frame_length

_mel_filters = mel_filter_bank(
    num_frequency_bins=1 + (fft_length // 2),
    num_mel_filters=num_mel_bins,
    min_frequency=0.0,
    max_frequency=8000.0,
    sampling_rate=16000,
    norm="slaney",
    mel_scale="slaney",
    triangularize_in_mel_space=False
)


class FeaturesExtractorModel(tf.Module):
    def __init__(self):
        super().__init__()

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=480000, dtype=tf.float32, name="normalized_audio"),
        ],
    )
    def serving(self, normalized_audio):
        stft_tensor = tf.signal.stft(normalized_audio, frame_length=frame_length, frame_step=frame_step,
                                     fft_length=fft_length, pad_end=True, window_fn=tf.signal.hann_window)
        magnitudes = tf.square(tf.abs(stft_tensor))

        mel_filters = tf.convert_to_tensor(_mel_filters, dtype=tf.float32)
        mel_spec = tf.tensordot(magnitudes, mel_filters, axes=1)

        mel_spec_safe = tf.clip_by_value(mel_spec, clip_value_min=1e-10, clip_value_max=31000.0)

        log_spec = tf.math.log(mel_spec_safe) / tf.math.log(tf.constant(10, dtype=tf.float32))
        log_spec = tf.math.maximum(log_spec, tf.math.reduce_max(log_spec) - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        log_spec = tf.transpose(log_spec)

        output = tf.expand_dims(log_spec, axis=0)

        return {"logmel": output}


r"""
Constructs a Whisper compatible FeaturesExtractorModel TF model and saves it to the temporary directory.
This save model can be used to convert it to the TFLite model.
        
Returns:
    `str`: The path to the saved model directory.
"""


def create_features_extractor() -> str:
    saved_model_dir = os.path.join(tempfile.gettempdir(), 'whisper2tflite', 'features_extractor')
    log.debug("features extractor creating start...")
    features_extractor_model = FeaturesExtractorModel()
    log.info("features extractor creating done")

    log.debug("features extractor saving start...")
    tf.saved_model.save(features_extractor_model, saved_model_dir,
                        signatures={"serving_default": features_extractor_model.serving})
    log.info("features extractor saving done")
    return saved_model_dir


if __name__ == "__main__":
    import argparse
    import convertor
    from config import default_model

    _cwd = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(_cwd)

    parser = argparse.ArgumentParser(description="The script creates Whisper features extractor")

    parser.add_argument("--artefacts_dir", type=str, help="The directory to save the model and assets",
                        default=os.path.join(_root, 'artefacts'))

    # Parse the arguments
    args = parser.parse_args()

    model_name = args.model_name
    artefacts_dir = args.artefacts_dir

    model_id = model_name.split("/")[-1]
    features_extractor_model_name = f"features-extractor.tflite"

    features_path = create_features_extractor()
    convertor.convert_saved(features_path, os.path.join(artefacts_dir, features_extractor_model_name), False)
