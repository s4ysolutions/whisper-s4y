import numpy as np
import tensorflow as tf

from tests import plot_input_features, plot_diff
from whisper_s4y.features_extractor import S4yFeaturesExtractor

atol = 0.82


def test_s4ymelmodule_callable(waveform_ar, transformers_input_features_ar):
    # Arrange
    module = S4yFeaturesExtractor()

    # Act
    output = module(waveform_ar)

    # Assert
    assert output.shape == (1, 80, 3000)
    assert output.dtype == tf.float32
    # for some reason the output is 1 frame ahead than the transformer output
    oif = output[:, :, :-1]
    tif = transformers_input_features_ar[:, :, 1:]

    try:
        np.testing.assert_allclose(oif.numpy(), tif.numpy(), rtol=1, atol=atol)
    except AssertionError as e:
        plot_diff(tif[0], oif[0], "diff model(ar1)")
        plot_input_features(tif, oif, "model(ar1)")
        raise e


def test_s4ymelmodule_serving(waveform_ar, transformers_input_features_ar):
    # Arrange
    module = S4yFeaturesExtractor()

    # Act
    output = module.serving(waveform_ar)['logmel']

    # Assert
    assert output.shape == (1, 80, 3000)
    assert output.dtype == tf.float32
    # for some reason the output is 1 frame ahead than the transformer output
    oif = output[:, :, :-1].numpy()
    tif = transformers_input_features_ar[:, :, 1:].numpy()

    try:
        np.testing.assert_allclose(oif, tif, rtol=1, atol=atol)
    except AssertionError as e:
        plot_diff(tif[0], oif[0], "diff model.serving(ar1)")
        plot_input_features(tif, oif, "model.serving(ar1)")
        raise e
