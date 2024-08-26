import numpy as np
import tensorflow as tf

from tests import test_model_id, test_log
from whisper_s4y.whisper import huggingface as hf
from whisper_s4y.whisper.huggingface.s4y_model import S4yDecoder


def decode_tflite(transformers_encoded_output, transformers_tokens, lang, optimize):
    # Arrange
    tflite_model_path = S4yDecoder(test_model_id, lang=lang, max_length=448).tflite(log=test_log, optimize=optimize)
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    runner = interpreter.get_signature_runner()

    # Act
    tokens1 = runner(encoder_hidden_states=transformers_encoded_output)['tokens']

    # Assert
    tokens1 = tf.constant(tokens1)
    tokens1.shape.assert_is_compatible_with([1, 448])

    first_eos = tf.argmax(tf.cast(tokens1 == 50257, tf.int32), axis=1).numpy()[0]
    tokens1 = tokens1[:, :first_eos + 1]

    tokenizer = hf.tokenizer(test_model_id)
    print(tokenizer.decode(tokens1[0]))
    print(tokenizer.decode(transformers_tokens[0]))

    np.testing.assert_equal(tokens1.numpy(), transformers_tokens.numpy())


# RuntimeError:
# tensorflow/lite/kernels/reduce.cc:445 std::apply(optimized_ops::Mean<T, U>, args) was not true.tensor
# flow/lite/kernels/reduce.cc:445 std::apply(optimized_ops::Mean<T, U>, args) was not true.tensor
# flow/lite/kernels/reduce.cc:445 std::apply(optimized_ops::Mean<T, U>, args) was not true.tensor
# flow/lite/kernels/reduce.cc:445 std::apply(optimized_ops::Mean<T, U>, args) was not true.tensor
# flow/lite/kernels/reduce.cc:445 std::apply(optimized_ops::Mean<T, U>, args) was not true.tensor
# flow/lite/kernels/reduce.cc:445 std::apply(optimized_ops::Mean<T, U>, args) was not true.tensor
# flow/lite/kernels/reduce.cc:445 std::apply(optimized_ops::Mean<T, U>, args) was not true.tensor
# flow/lite/kernels/reduce.cc:445 std::apply(optimized_ops::Mean<T, U>, args) was not true.tensor
# flow/lite/kernels/reduce.cc:445 std::apply(optimized_ops::Mean<T, U>, args) was not true.tensor
# flow/lite/kernels/reduce.cc:445 std::apply(optimized_ops::Mean<T, U>, args) was not true.tensor
# flow/lite/kernels/reduce.cc:445 std::apply(optimized_ops::Mean<T, U>, args) was not true.tensor
# flow/lite/kernels/reduce.cc:445 std::apply(optimized_ops::Mean<T, U>, args) was not true.tensor
# flow/lite/kernels/reduce.cc:445 std::apply(optimized_ops::Mean<T, U>, args) was not true.tensor
# flow/lite/kernels/reduce.cc:445 std::apply(optimized_ops::Mean<T, U>, args) was not true.tensor
# flow/lite/kernels/reduce.cc:445 std::apply(optimized_ops::Mean<T, U>, args) was not true.tensor
# flow/lite/kernels/reduce.cc:445 std::apply(optimized_ops::Mean<T, U>, args) was not true.gather
# index out of boundsNode number 33 (GATHER) failed to invoke.Node number 189 (WHILE) failed to invoke.
def test_decode_tflite_ar_not_optimize(transformers_encoded_output_ar, tokens_ar):
    decode_tflite(transformers_encoded_output_ar, tokens_ar, lang='ar', optimize=False)


def test_decode_tflite_ar_optimize(transformers_encoded_output_ar, tokens_ar):
    decode_tflite(transformers_encoded_output_ar, tokens_ar, lang='ar', optimize=True)


def test_decode_tflite_en_not_optimize(transformers_encoded_output_en, tokens_en):
    decode_tflite(transformers_encoded_output_en, tokens_en, lang='en', optimize=False)


def test_decode_tflite_en_optimize(transformers_encoded_output_en, tokens_en):
    decode_tflite(transformers_encoded_output_en, tokens_en, lang='en', optimize=True)
