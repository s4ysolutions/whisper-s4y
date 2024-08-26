import tensorflow as tf
from tests import test_model_id, test_log
from whisper_s4y.whisper import huggingface as hf

from whisper_s4y.whisper.huggingface.s4y_model import S4yPcmEncoder, S4yDecoder


def test_en_with_pcm_encoder_call(waveform_en, transcription_en):
    # Arrange
    encoder = S4yPcmEncoder(test_model_id)
    decoder = S4yDecoder(test_model_id, lang="en", max_length=448)
    tokenizer = hf.tokenizer(test_model_id)

    # Act
    encoded = encoder(normalized_audio=waveform_en)
    tokens = decoder(encoded.last_hidden_state)
    transcription = tokenizer.decode(tokens[0])

    # Assert
    assert transcription == transcription_en


def test_en_with_pcm_encoder_tflite(waveform_en, transcription_en):
    # Arrange
    tflite_model_path = S4yPcmEncoder(test_model_id).tflite(log=test_log)
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    encoder = interpreter.get_signature_runner()

    decoder = S4yDecoder(test_model_id, lang="en", max_length=448)
    tokenizer = hf.tokenizer(test_model_id)

    # Act
    encoded = encoder(normalized_audio=waveform_en)
    tokens = decoder(encoded['last_hidden_state'])
    transcription = tokenizer.decode(tokens[0])

    # Assert
    assert transcription == transcription_en

# def xtest_ar_with_pcm_encoder_call(waveform_en, forced_decoder_ids_en, transcription_en):
#    # Arrange
#    encoder = S4yPcmEncoder(test_model_id)
#    decoder = S4yDecoder(test_model_id, lang="en", max_length=448)
#    tokenizer = hf.tokenizer(test_model_id)
#
#    # Act
#    encoded = encoder(waveform_en)
#    tokens = decoder(encoded.last_hidden_state)
#    transcription = tokenizer.decode(tokens[0])
#
#    # Assert
#    transformers_whisper = hf.for_conditional_generation(test_model_id)
#    transformers_processor = hf.processor(test_model_id)
#    # transformers_features_extractor = hf.feature_extractor(transformers_processor)
#    transformers_tokenizer = hf.tokenizer(transformers_processor)
#
#    transformers_input_features = transformers_processor(waveform_en).input_features
#    transformers_tokens = transformers_whisper.generate(transformers_input_features,
#                                                        forced_decoder_ids=forced_decoder_ids_en)
#    transformers_transcription = transformers_tokenizer.decode(transformers_tokens[0])
#
#    print(transcription)
#    print(transformers_transcription)
#
#    assert transcription == transformers_transcription
#
##    transformers = transformer_encoded_output_ar.numpy()
##    try:
##        np.testing.assert_allclose(output, transformers, rtol=1, atol=14)
##    except AssertionError as e:
##        plot_diff(transformers[0], output[0], "diff pcm_encoder serving(ar1)")
##        plot_encoded_output(transformers, output, "pcm encoder(ar1)")
##        raise e
#
