import numpy as np

from whisper_s4y.whisper.huggingface.s4y_model import S4yPcmEncoder
from tests import plot_encoded_output, test_model_id, plot_diff

atol=1

def test_pcm_encoder_call(transformers_encoded_output_ar, waveform_ar):
    # Arrange
    module = S4yPcmEncoder(test_model_id)

    # Act
    output = module(waveform_ar)
    output = output.last_hidden_state

    # Assert
    transformers = transformers_encoded_output_ar.numpy()
    try:
        np.testing.assert_allclose(output, transformers, rtol=1, atol=atol)
    except AssertionError as e:
        plot_diff(transformers[0], output[0], "diff pcm_encoder serving(ar1)")
        plot_encoded_output(transformers, output, "pcm encoder(ar1)")
        raise e


def test_encoder_serving(transformers_encoded_output_ar, waveform_ar):
    # Arrange
    module = S4yPcmEncoder(test_model_id)

    # Act
    output = module.serving(waveform_ar)['last_hidden_state']

    # Assert
    o = output
    t = transformers_encoded_output_ar

    try:
        np.testing.assert_allclose(o.numpy(), t.numpy(), rtol=1, atol=atol)
    except AssertionError as e:
        plot_diff(t[0], o[0], "diff pcm_encoder serving(ar1)")
        plot_encoded_output(t, o, "pcm_encoder serving(ar1)")
        raise e
