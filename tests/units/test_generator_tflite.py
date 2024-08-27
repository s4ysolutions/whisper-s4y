import numpy as np
import tensorflow as tf
from tests import test_model_id, test_log
from whisper_s4y.whisper.huggingface.s4y_model import S4yGenerator
from whisper_s4y.whisper import huggingface as hf


def generate_tflite(transformers_input_features, transformers_tokens, lang, optimize):
    # Arrange
    tflite_model_path = S4yGenerator(test_model_id, lang=lang).tflite(log=test_log, optimize=optimize)
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    runner = interpreter.get_signature_runner()

    # Act
    tokens1 = runner(input_features=transformers_input_features)["sequences"]

    tokenizer = hf.tokenizer(test_model_id)
    print(tokenizer.decode(tokens1[0]))
    print(tokenizer.decode(transformers_tokens[0]))
    # Assert
    tokens1 = tf.constant(tokens1)
    tokens1.shape.assert_is_compatible_with([1, 448])

    first_eos = tf.argmax(tf.cast(tokens1 == 50257, tf.int32), axis=1).numpy()[0]
    tokens1 = tokens1[:, :first_eos + 1]

    np.testing.assert_equal(tokens1.numpy(), transformers_tokens.numpy())


# E     RuntimeError:
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
# index out of boundsNode number 33 (GATHER) failed to invoke.Node number 390 (WHILE) failed to invoke.
def test_generate_ar_tflite_non_optimize(transformers_input_features_ar, tokens_ar):
    generate_tflite(transformers_input_features_ar, tokens_ar, lang='ar', optimize=False)


def test_generate_ar_tflite_optimize(transformers_input_features_ar, tokens_ar):
    generate_tflite(transformers_input_features_ar, tokens_ar, lang='ar', optimize=True)


def test_generate_en_tflite_not_optimize(transformers_input_features_en, tokens_en):
    generate_tflite(transformers_input_features_en, tokens_en, lang='en', optimize=False)

def test_generate_en_tflite_optimize(transformers_input_features_en, tokens_en):
    generate_tflite(transformers_input_features_en, tokens_en, lang='en', optimize=True)
