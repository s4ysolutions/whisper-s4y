import numpy as np
import tensorflow as tf
from tests import test_model_id

from whisper_s4y.whisper import huggingface as hf
from whisper_s4y.whisper.huggingface.s4y_model import S4yEncoderDecoder


def test_encoder_decoder_call_ar(transformers_input_features_ar, tokens_ar):
    # Arrange
    model = S4yEncoderDecoder(test_model_id, lang='ar')

    # Act
    tokens1 = model(input_features=transformers_input_features_ar)

    tokenizer = hf.tokenizer(test_model_id)
    print(tokenizer.decode(tokens1[0]))
    print(tokenizer.decode(tokens_ar[0]))

    # Assert
    np.testing.assert_equal(tokens1.numpy(), tokens_ar.numpy())


def test_encoder_decoder_serving_ar(transformers_input_features_ar, tokens_ar):
    # Arrange
    model = S4yEncoderDecoder(test_model_id, lang='ar')

    # Act
    tokens1 = model.serving(input_features=transformers_input_features_ar)['tokens']

    tokenizer = hf.tokenizer(test_model_id)
    print(tokenizer.decode(tokens1[0]))
    print(tokenizer.decode(tokens_ar[0]))

    # Assert
    tokens1.shape.assert_is_compatible_with([1, 448])

    first_eos = tf.argmax(tf.cast(tokens1 == 50257, tf.int32), axis=1).numpy()[0]
    tokens1 = tokens1[:, :first_eos + 1]

    tf.debugging.assert_equal(tokens1, tokens_ar)