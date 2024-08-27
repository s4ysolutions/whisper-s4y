import logging
import os
import sys
import tensorflow as tf

from whisper_s4y.features_extractor.lib import mel_filter_bank
from whisper_s4y import tflite as _tflite, log as _log

FRAME_LENGTH = 400
FRAME_STEP = 160
num_mel_bins = 80
FFT_LENGTH = 512  # 2^9 > frame_length

_mel_filters = mel_filter_bank(
    num_frequency_bins=1 + (FFT_LENGTH // 2),
    # num_frequency_bins=FRAME_LENGTH // 2 + 1,
    num_mel_filters=num_mel_bins,
    min_frequency=0.0,
    max_frequency=8000.0,
    sampling_rate=16000,
    norm="slaney",
    mel_scale="slaney",
    triangularize_in_mel_space=False
)


def window_function(window_length, dtype, name=None):
    return tf.signal.hann_window(window_length, dtype=dtype, name=name)


class S4yFeaturesExtractor(_tflite.ServingModel):
    def __init__(self):
        super().__init__()

    def call(self, normalized_audio):
        stft_tensor = tf.signal.stft(normalized_audio, frame_length=FRAME_LENGTH, frame_step=FRAME_STEP,
                                     fft_length=FFT_LENGTH, pad_end=True, window_fn=window_function)
        # cut off everything beyond widnow, see test_stft.py
        # stft_tensor = stft_tensor[:, :1 + (FRAME_LENGTH // 2)]
        magnitudes = tf.abs(stft_tensor) ** 2

        mel_filters = tf.convert_to_tensor(_mel_filters, dtype=tf.float32)
        mel_spec = tf.tensordot(magnitudes, mel_filters, axes=1)

        mel_spec_safe = tf.clip_by_value(mel_spec, clip_value_min=1e-10, clip_value_max=31000.0)

        log_spec = tf.math.log(mel_spec_safe) / tf.math.log(tf.constant(10, dtype=tf.float32))
        log_spec = tf.math.maximum(log_spec, tf.math.reduce_max(log_spec) - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        log_spec = tf.transpose(log_spec)

        output = tf.expand_dims(log_spec, axis=0)

        return output

    def __call__(self, normalized_audio):
        return self.call(normalized_audio)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=480000, dtype=tf.float32, name="normalized_audio"),
        ],
    )
    def serving(self, normalized_audio):
        return {'logmel': self.call(normalized_audio)}

    def tflite(self, model_name='features-extractor', tflite_model_path=None,
               log=_log, optimize=False) -> str:
        return _tflite.tflite(model_name, self, tflite_model_path=tflite_model_path, optimize=optimize,
                              log=log)


if __name__ == "__main__":
    log = logging.getLogger(os.path.basename(os.path.dirname(__file__)))
    log.addHandler(logging.StreamHandler(sys.stdout))
    log.setLevel(logging.DEBUG)

    S4yFeaturesExtractor().tflite(log=log)
