import numpy as np
import tensorflow as tf

from whisper_s4y.whisper import huggingface as hf
from whisper_s4y.whisper.huggingface.s4y_model import S4yEncoderDecoder
from tests import plot_encoded_output, test_model_id, test_log, plot_diff


def encoder_decoder_tflite(transformer_input_features, tokens, lang: str, optimize: bool, model_id=test_model_id):
    # Arrange
    tflite_model_path = S4yEncoderDecoder(model_id, lang=lang).tflite(log=test_log, optimize=optimize)
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    runner = interpreter.get_signature_runner()

    # Act
    tokens1 = runner(input_features=transformer_input_features)['tokens']

    tokenizer = hf.tokenizer(test_model_id)
    print(tokenizer.decode(tokens1[0]))
    print(tokenizer.decode(tokens[0]))
    # Assert
    try:
        np.testing.assert_equal(tokens1.numpy(), tokens.numpy())
    except AssertionError as e:
        plot_diff(tokens1[0], tokens[0], f"diff encoder tflite/optimized={optimize} ({lang})")
        plot_encoded_output(tokens1, tokens, f"encoder tflite/optimized={optimize} ({lang})")
        raise e


def test_encoder_decode_tflite_ar_optimize(transformers_input_features_ar, tokens_ar):
    encoder_decoder_tflite(transformers_input_features_ar, tokens_ar, 'ar', optimize=True)


def test_encoder_decode_tflite_ar_non_optimize(transformers_input_features_ar, tokens_ar):
    encoder_decoder_tflite(transformers_input_features_ar, tokens_ar, 'ar', optimize=False)


def test_encoder_decode_tflite_en_optimize(transformers_input_features_en, tokens_en):
    encoder_decoder_tflite(transformers_input_features_en, tokens_en, 'en', optimize=True)


def test_encoder_decode_tflite_en_optimize_base(transformers_input_features_en, tokens_en):
    encoder_decoder_tflite(transformers_input_features_en, tokens_en, 'en', optimize=True,
                           model_id='openai/whisper-base')


def test_encoder_decode_tflite_en_non_optimize(transformers_input_features_en, tokens_en):
    encoder_decoder_tflite(transformers_input_features_en, tokens_en, 'en', optimize=False)
