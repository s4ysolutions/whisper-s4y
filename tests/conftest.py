import numpy as np
import pickle
import pytest
import tensorflow as tf

from tests import normalized_audio_from_wav, transformers_input_features, transformers_encoded_output, test_model_id, \
    test_data_path
from whisper_s4y.whisper import huggingface as hf


@pytest.fixture
def waveform_ar():
    return normalized_audio_from_wav("ar/1-1.wav")


@pytest.fixture
def waveform_en():
    return normalized_audio_from_wav("en/OSR_us_000_0030_16k.wav")


@pytest.fixture
def transformers_input_features_ar(waveform_ar):
    return transformers_input_features(waveform_ar)


@pytest.fixture
def transformers_input_features_en(waveform_en):
    return transformers_input_features(waveform_en)


@pytest.fixture
def transformers_encoded_output_ar(transformers_input_features_ar):
    return transformers_encoded_output(transformers_input_features_ar)


@pytest.fixture
def transformers_encoded_output_en(transformers_input_features_en):
    return transformers_encoded_output(transformers_input_features_en)


@pytest.fixture
def decoder_iter_1_kwargs_ar(transformers_encoded_output_ar):
    return dict(
        input_ids=tf.constant([[50258]], dtype=tf.int32),
        position_ids=tf.constant([[0]], dtype=tf.int32),
        encoder_hidden_states=transformers_encoded_output_ar,
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )


@pytest.fixture
def decoder_iter_1_output_ar(decoder_iter_1_kwargs_ar):
    return hf.decoder(test_model_id)(**decoder_iter_1_kwargs_ar)


@pytest.fixture
def transformer_output_ar(transformers_input_features_ar, forced_decoder_ids_ar):
    return hf.for_conditional_generation(test_model_id).generate(transformers_input_features_ar,
                                                                 forced_decoder_ids=forced_decoder_ids_ar)


@pytest.fixture
def forced_decoder_ids_ar():
    return hf.tokenizer(test_model_id).get_decoder_prompt_ids(language="ar", task="transcribe")


@pytest.fixture
def forced_decoder_ids_en():
    return hf.tokenizer(test_model_id).get_decoder_prompt_ids(language="en", task="transcribe")


@pytest.fixture
def transcription_ar():
    return '<|startoftranscript|><|ar|><|transcribe|><|notimestamps|> أعود بالله من الشيطوني رجم بس بالله رحمان رحم المهمد للله رب العالمين<|endoftext|>'


@pytest.fixture
def transcription_ar_pcm_encoder():
    return '<|startoftranscript|><|ar|><|transcribe|><|notimestamps|> أعود بالله من الشيطوني رجم بس بالله رحمان رحم<|endoftext|>'


@pytest.fixture
def transcription_en():
    return '<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Paint the sockets in the wall, dull green. The child crawled into the dense grass, bribes fail where honest men work. Trampal the spark, else the flames will spread. The hilt of the sword was carved with fine designs. A round hole was drilled through the thin board. Footprints showed the path he took up the beach.<|endoftext|>'


@pytest.fixture
def tokens_ar():
    with open(test_data_path("ar/tokens_1-1_ar.pkl"), "rb") as f:
        tokens = pickle.load(f)
    return tf.convert_to_tensor(tokens)


@pytest.fixture
def tokens_ar_translation():
    return tf.constant(
        [[50258, 50259, 50359, 50363, 286, 486, 312, 5404, 281, 312, 294, 264, 1315, 295, 4574, 13, 50257]],
        dtype=tf.int32)


@pytest.fixture
def tokens_en(transformers_input_features_en, forced_decoder_ids_en):
    tokens = (hf.for_conditional_generation(test_model_id).generate(transformers_input_features_en,
                                                                    forced_decoder_ids=forced_decoder_ids_en))
    return tokens


@pytest.fixture()
def wfcg_outputs_logits_1_ar():
    numpy = np.load(test_data_path("ar/wfcg_outputs_logits_1_ar.npy"))
    return tf.convert_to_tensor(numpy)


@pytest.fixture
def next_tokens_scores_1_ar():
    numpy = np.load(test_data_path("ar/next_tokens_scores_1_ar.npy"))
    return tf.convert_to_tensor(numpy)


@pytest.fixture
def next_tokens_1_ar():
    numpy = np.load(test_data_path("ar/next_tokens_1_ar.npy"))
    return tf.convert_to_tensor(numpy)


@pytest.fixture
def generated_1():
    tensor = tf.fill([1, 448], 50257)
    tensor = tf.tensor_scatter_nd_update(tensor, [[0, 0]], [50258])
    return tensor


@pytest.fixture
def finished_sequences_1():
    return tf.constant([False], dtype=tf.bool)


@pytest.fixture
def eos_token_id():
    return 50257


@pytest.fixture
def pad_token_id():
    return 50257
