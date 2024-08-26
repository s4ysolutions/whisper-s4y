import numpy as np
import tensorflow as tf

from whisper_s4y.whisper.huggingface.s4y_model import S4yEncoder
from tests import plot_encoded_output, test_model_id, test_log, plot_diff


def encoder_tflite(transformer_encoded_output_ar, transformer_input_features_ar, optimize: bool):
    # Arrange
    tflite_model_path = S4yEncoder(test_model_id).tflite(log=test_log, optimize=optimize)
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    runner = interpreter.get_signature_runner()

    # Act
    output = runner(input_features=transformer_input_features_ar)['last_hidden_state']

    # Assert
    o = output
    t = transformer_encoded_output_ar.numpy()

    try:
        atol = 1e-20
        np.testing.assert_allclose(o, t, rtol=1, atol=atol)
    except AssertionError as e:
        plot_diff(t[0], o[0], f"diff encoder tflite/optimized={optimize} (ar1)")
        plot_encoded_output(t, o, f"encoder tflite/optimized={optimize} (ar1)")
        raise e


def test_encoder_tflite_non_optimize(transformers_encoded_output_ar, transformers_input_features_ar):
    encoder_tflite(transformers_encoded_output_ar, transformers_input_features_ar, optimize=False)


def test_encoder_tflite_optimize(transformers_encoded_output_ar, transformers_input_features_ar):
    encoder_tflite(transformers_encoded_output_ar, transformers_input_features_ar, optimize=True)


def test_encoder_tflite_optimize(transformers_encoded_output_ar, transformers_input_features_ar):
    encoder_tflite(transformers_encoded_output_ar, transformers_input_features_ar, optimize=True)
