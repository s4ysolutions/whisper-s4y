import numpy as np

from whisper_s4y.whisper.huggingface.s4y_model import S4yEncoder
from tests import plot_encoded_output, test_model_id, plot_diff


def test_encoder_call(transformers_encoded_output_ar, transformers_input_features_ar):
    # Arrange
    module = S4yEncoder(test_model_id)

    # Act
    output = module(transformers_input_features_ar)
    output = output.last_hidden_state

    # Assert
    transformers = transformers_encoded_output_ar.numpy()
    try:
        np.testing.assert_equal(output, transformers)
    except AssertionError as e:
        plot_encoded_output(transformers, output, "encoder(ar1)")
        raise e


def test_encoder_serving(transformers_encoded_output_ar, transformers_input_features_ar):
    # Arrange
    module = S4yEncoder(test_model_id)

    # Act
    output = module.serving(transformers_input_features_ar)['last_hidden_state']

    # Assert
    o = output
    t = transformers_encoded_output_ar

    try:
        np.testing.assert_allclose(o.numpy(), t.numpy(), rtol=1, atol=1e-20)
        #np.testing.assert_equal(o.numpy(), t.numpy())
    except AssertionError as e:
        plot_diff(t[0], o[0], "diff encoder serving(ar1)")
        plot_encoded_output(t, o, "encoder serving(ar1)")
        raise e
