import numpy as np
import tensorflow as tf

from whisper_s4y.whisper.huggingface.s4y_model import S4yPcmEncoder
from tests import plot_encoded_output, test_model_id, test_log, plot_diff


def pcm_encoder_tflite(transformers_encoded_output, waveform, optimize=True):
    # Arrange
    tflite_model_path = S4yPcmEncoder(test_model_id).tflite(log=test_log, optimize=optimize)
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    runner = interpreter.get_signature_runner()

    # Act
    output = runner(normalized_audio=waveform)['last_hidden_state']

    # Assert
    o = output
    t = transformers_encoded_output.numpy()

    try:
        np.testing.assert_allclose(o, t, rtol=1, atol=1)
    except AssertionError as e:
        plot_diff(t[0], o[0], f"diff pcm_encoder tflite optimized={optimize}")
        plot_encoded_output(t, o, f"pcm_encoder tflite optimized={optimize}")
        raise e


def test_pcm_encoder_tflite_optimized(transformers_encoded_output_ar, waveform_ar):
    pcm_encoder_tflite(transformers_encoded_output_ar, waveform_ar, optimize=True)


def test_pcm_encoder_tflite_not_optimized(transformers_encoded_output_ar, waveform_ar):
    pcm_encoder_tflite(transformers_encoded_output_ar, waveform_ar, optimize=False)
