import numpy as np
import tensorflow as tf
from tests import test_model_id
from whisper_s4y.whisper.huggingface.s4y_model import S4yDecoder


def test_decoder_ar_callable(transformers_encoded_output_ar, tokens_ar):
    # Arrange
    model = S4yDecoder(test_model_id, lang='ar')

    # Act
    output1 = model(encoder_hidden_states=transformers_encoded_output_ar)
    output2 = model(encoder_hidden_states=transformers_encoded_output_ar)

    # Assert
    np.testing.assert_equal(output1.numpy(), tokens_ar.numpy())
    np.testing.assert_equal(output2.numpy(), tokens_ar.numpy())


def test_s4y_ar_serving(transformers_encoded_output_ar, tokens_ar):
    # Arrange
    model = S4yDecoder(test_model_id, lang='ar')

    # Act
    tokens1 = model.serving(encoder_hidden_states=transformers_encoded_output_ar)['tokens']
    tokens2 = model.serving(encoder_hidden_states=transformers_encoded_output_ar)['tokens']

    # Assert
    tokens1.shape.assert_is_compatible_with([1, 448])
    tokens2.shape.assert_is_compatible_with([1, 448])

    first_eos = tf.argmax(tf.cast(tokens1 == 50257, tf.int32), axis=1).numpy()[0]
    tokens1 = tokens1[:, :first_eos + 1]
    first_eos = tf.argmax(tf.cast(tokens2 == 50257, tf.int32), axis=1).numpy()[0]
    tokens2 = tokens2[:, :first_eos + 1]

    tf.debugging.assert_equal(tokens1, tokens_ar)
    tf.debugging.assert_equal(tokens2, tokens_ar)
