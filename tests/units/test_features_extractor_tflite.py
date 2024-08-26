import numpy as np
import tensorflow as tf

from tests import plot_input_features, test_log, plot_diff
from whisper_s4y.features_extractor import S4yFeaturesExtractor


def s4ymelmodule_tflite(waveform, transformers_input_features, optimized=True):
    # Arrange
    tflite_model_path = S4yFeaturesExtractor().tflite(log=test_log, optimize=optimized)
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    runner = interpreter.get_signature_runner()

    # Act
    output = runner(normalized_audio=waveform)['logmel']

    # Assert
    assert output.shape == (1, 80, 3000)
    assert output.dtype == tf.float32
    # for some reason the output is 1 frame ahead than the transformer output
    oif = output[:, :, :-1]
    tif = transformers_input_features[:, :, 1:].numpy()

    try:
        np.testing.assert_allclose(oif, tif, rtol=1, atol=0.1)
    except AssertionError as e:
        plot_diff(tif[0], oif[0], f"diff model.serving optimized={optimized}")
        plot_input_features(tif, oif, "tflite optimized={optimized}")
        raise e


def test_s4ymelmodule_tflite_optimized(transformers_input_features_ar, waveform_ar):
    s4ymelmodule_tflite(waveform_ar, transformers_input_features_ar, optimized=True)


def test_s4ymelmodule_tflite_not_optimized(transformers_input_features_ar, waveform_ar):
    s4ymelmodule_tflite(waveform_ar, transformers_input_features_ar, optimized=False)
