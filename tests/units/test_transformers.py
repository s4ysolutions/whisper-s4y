import tensorflow as tf

from transformers import TFLogitsProcessorList
from tests import test_model_id
from whisper_s4y.whisper import huggingface as hf


def test_for_conditional_generation_ar(transformers_input_features_ar, forced_decoder_ids_ar, tokens_ar,
                                       transcription_ar):
    # Arrange
    model = hf.for_conditional_generation(test_model_id)
    tokenizer = hf.tokenizer(test_model_id)
    # model.config.forced_decoder_ids = forced_decoder_ids_ar

    # Act
    tokens = model.generate(transformers_input_features_ar, forced_decoder_ids=forced_decoder_ids_ar)
    output = tokenizer.decode(tokens[0])

    # Assert
    tf.debugging.assert_equal(tokens, tokens_ar)
    assert output == transcription_ar


def test_prepare_inputs_for_generation_ar(transformers_encoded_output_ar):
    # Arrange
    gm = hf.for_conditional_generation(test_model_id)
    # Act
    model_inputs = gm.prepare_inputs_for_generation(
        decoder_input_ids=tf.constant([[50258]], dtype=tf.int32),
        use_cache=False,
        encoder_outputs=transformers_encoded_output_ar,
    )
    # Assert
    assert len(model_inputs) == 7
    assert model_inputs["input_features"] is None
    tf.debugging.assert_equal(model_inputs["encoder_outputs"], transformers_encoded_output_ar)
    assert model_inputs["past_key_values"] is None
    tf.debugging.assert_equal(model_inputs["decoder_input_ids"], tf.constant([[50258]], dtype=tf.int32))
    assert model_inputs["use_cache"] is False
    assert model_inputs["decoder_attention_mask"] is None
    tf.debugging.assert_equal(model_inputs["decoder_position_ids"], tf.constant([[0]], dtype=tf.int32))


def test_get_logits_processor(forced_decoder_ids_ar):
    # Arrange
    gm = hf.for_conditional_generation(test_model_id)
    # gm.config.forced_decoder_ids_ar = forced_decoder_ids_ar
    # generation_config is copied from the model config
    gm.generation_config.forced_decoder_ids = forced_decoder_ids_ar
    logits_processor = TFLogitsProcessorList()
    input_ids_seq_length = 1
    # Act
    logits_processor = gm._get_logits_processor(
        generation_config=gm.generation_config,
        input_ids_seq_length=input_ids_seq_length,
        logits_processor=logits_processor,
    )
    # Assert
    assert len(logits_processor) == 3
    assert logits_processor[0].__class__.__name__ == "TFSuppressTokensLogitsProcessor"
    assert logits_processor[1].__class__.__name__ == "TFSuppressTokensAtBeginLogitsProcessor"
    assert logits_processor[2].__class__.__name__ == "TFForceTokensLogitsProcessor"
    tf.debugging.assert_equal(logits_processor[2].force_token_array,
                              tf.constant([-1, forced_decoder_ids_ar[0][1], forced_decoder_ids_ar[1][1],
                                           forced_decoder_ids_ar[2][1]], dtype=tf.int32))


def test_model_output_first_ar(transformers_encoded_output_ar, wfcg_outputs_logits_1_ar):
    # Arrange
    gm = hf.for_conditional_generation(test_model_id)
    # Act
    model_outputs = gm(
        input_features=None,
        encoder_outputs=transformers_encoded_output_ar,
        past_key_values=None,
        decoder_input_ids=tf.constant([[50258]], dtype=tf.int32),
        use_cache=False,
        decoder_attention_mask=None,
        decoder_position_ids=tf.constant([[0]], dtype=tf.int32),
    )
    # Assert
    tf.debugging.assert_equal(model_outputs["logits"], wfcg_outputs_logits_1_ar)


def test_greedy_search_ar(tokens_ar, transformers_encoded_output_ar, forced_decoder_ids_ar, transcription_ar):
    # Arrange
    gm = hf.for_conditional_generation(test_model_id)
    tokenizer = hf.tokenizer(test_model_id)

    gm.generation_config.forced_decoder_ids = forced_decoder_ids_ar

    input_ids = tf.constant([[50258]], dtype=tf.int32)
    input_ids_seq_length = 1

    logits_processor = TFLogitsProcessorList()
    logits_processor = gm._get_logits_processor(
        generation_config=gm.generation_config,
        input_ids_seq_length=input_ids_seq_length,
        logits_processor=logits_processor,
    )
    # Act
    tokens = gm.greedy_search(
        input_ids=input_ids,
        max_length=56,
        pad_token_id=50257,
        eos_token_id=50257,
        logits_processor=logits_processor,
        output_attentions=False,
        output_hidden_states=False,
        output_scores=False,
        return_dict_in_generate=False,
        use_cache=True,
        encoder_outputs=transformers_encoded_output_ar,
    )
    output = tokenizer.decode(tokens[0])
    # Assert

    tf.debugging.assert_equal(tokens, tokens_ar)
    assert output == transcription_ar
