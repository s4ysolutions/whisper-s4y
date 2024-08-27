import numpy as np
import tensorflow as tf
from tests import test_model_id
from whisper_s4y.whisper.huggingface.s4y_model import S4yGenerator
from whisper_s4y.whisper import huggingface as hf


def test_generate_ar_callable(transformers_input_features_ar, tokens_ar):
    # Arrange
    model = S4yGenerator(test_model_id, lang='ar')

    # Act
    output1 = model(input_features=transformers_input_features_ar)["sequences"]

    tokenizer = hf.tokenizer(test_model_id)
    print(tokenizer.decode(output1[0]))
    print(tokenizer.decode(tokens_ar[0]))
    # Assert
    np.testing.assert_equal(output1.numpy(), tokens_ar.numpy())


def test_s4y_ar_serving(transformers_input_features_ar, tokens_ar):
    # Arrange
    model = S4yGenerator(test_model_id, lang='ar')

    # Act
    tokens1 = model.serving(input_features=transformers_input_features_ar)["sequences"]

    tokenizer = hf.tokenizer(test_model_id)
    print(tokenizer.decode(tokens1[0]))
    print(tokenizer.decode(tokens_ar[0]))
    # Assert
    tokens1.shape.assert_is_compatible_with([1, 448])

    first_eos = tf.argmax(tf.cast(tokens1 == 50257, tf.int32), axis=1).numpy()[0]
    tokens1 = tokens1[:, :first_eos + 1]

    tf.debugging.assert_equal(tokens1, tokens_ar)


def test_generate_en_callable(transformers_input_features_en, tokens_en):
    # Arrange
    model = S4yGenerator(test_model_id, lang='en')

    # Act
    output1 = model(input_features=transformers_input_features_en)["sequences"]

    tokenizer = hf.tokenizer(test_model_id)
    print(tokenizer.decode(output1[0]))
    print(tokenizer.decode(tokens_en[0]))
    # Assert
    np.testing.assert_equal(output1.numpy(), tokens_en.numpy())


def test_generate_en_serving(transformers_input_features_en, tokens_en):
    # Arrange
    model = S4yGenerator(test_model_id, lang='en')

    # Act
    tokens1 = model.serving(input_features=transformers_input_features_en)["sequences"]

    tokenizer = hf.tokenizer(test_model_id)
    print(tokenizer.decode(tokens1[0]))
    print(tokenizer.decode(tokens_en[0]))

    # Assert
    first_eos = tf.argmax(tf.cast(tokens1 == 50257, tf.int32), axis=1).numpy()[0]
    tokens1 = tokens1[:, :first_eos + 1]
    np.testing.assert_equal(tokens1.numpy(), tokens_en.numpy())
