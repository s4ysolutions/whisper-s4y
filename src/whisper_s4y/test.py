import argparse
import os.path
import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from transformers import TFWhisperForConditionalGeneration, WhisperTokenizer, WhisperFeatureExtractor, WhisperConfig

from whisper_s4y import log


def normalized_audio_from_wav(wav_file_path) -> tf.TensorSpec(shape=[480000], dtype=tf.float32):
    waveform, sample_rate = tf.audio.decode_wav(tf.io.read_file(wav_file_path))
    if sample_rate != 16000:
        print(f"sample rate is {sample_rate}, resampling...")
        waveform = tfio.audio.resample(waveform, rate_in=sample_rate, rate_out=16000)
        print("ok")
    audio = waveform[:, 0]
    size = audio.shape[0]
    if size < 480000:
        audio = tf.concat([audio, tf.zeros(480000 - size)], 0)
    elif size > 480000:
        audio = audio[:480000]
    return audio


def tflite_generator(tflite_model_path: str):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    runner = interpreter.get_signature_runner()

    def _(input_features: tf.TensorSpec(shape=[1, 80, 3000], dtype=tf.float32)) -> tf.TensorSpec(shape=[1, 128],
                                                                                                 dtype=tf.int32):
        return tf.convert_to_tensor(runner(input_features=input_features)['sequences'])

    return _


def tflite_features_extractor(tflite_model_path: str):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    runner = interpreter.get_signature_runner()

    def _(audio: tf.TensorSpec(shape=[480000], dtype=tf.float32)) -> tf.TensorSpec(shape=[1, 80, 3000],
                                                                                   dtype=tf.float32):
        return tf.convert_to_tensor(runner(normalized_audio=audio)['logmel'])

    return _


def transformers_generator(model_name: str, lang):
    model = TFWhisperForConditionalGeneration.from_pretrained(model_name)
    tokenizer = WhisperTokenizer.from_pretrained(model_name)

    def _(input_features: tf.TensorSpec(shape=[1, 80, 3000], dtype=tf.float32)) -> tf.TensorSpec(shape=[1, 128],
                                                                                                 dtype=tf.int32):
        return model.generate(input_features,
                              forced_decoder_ids=tokenizer.get_decoder_prompt_ids(language=lang, task="transcribe"),
                              return_dict_in_generate=True)['sequences']

    return _


def transformers_features_extractor(model_name: str):
    extractor = WhisperFeatureExtractor.from_pretrained(model_name)

    def _(audio: tf.TensorSpec(shape=[480000], dtype=tf.float32)) -> tf.TensorSpec(shape=[1, 80, 3000],
                                                                                   dtype=tf.float32):
        return extractor(audio, sampling_rate=16000, return_tensors="tf")['input_features']

    return _


def transformers_decoder(model_name):
    tokenizer = WhisperTokenizer.from_pretrained(model_name)

    def _(tokens: tf.Tensor) -> str:
        return tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]

    return _


def audio_ar_1():
    return normalized_audio_from_wav(os.path.join(_project_root, 'test_data', 'ar', '1-1.wav')), 'ar 1'


def audio_ar_2():
    return normalized_audio_from_wav(os.path.join(_project_root, 'test_data', 'ar', 'al-fatiha.wav')), 'ar 2'


def audio_en_1():
    return normalized_audio_from_wav(os.path.join(_project_root, 'test_data', 'en', 'OSR_us_000_0030_16k.wav')), 'en 1'


def audio_en_2():
    return normalized_audio_from_wav(os.path.join(_project_root, 'test_data', 'en', 'harvard-16k.wav')), 'en 2'


def plot_audio(audio: tf.TensorSpec(shape=[480000], dtype=tf.float32), ax: Axes, title: str = 'Audio'):
    time = [float(n) / 16000 for n in range(len(audio))]
    ax.plot(time, audio)
    ax.set_title(title)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Amplitude')


def plot_logmel(logmel: tf.TensorSpec(shape=[80, 3000], dtype=tf.float32), fig, ax: Axes, title: str = 'LogMel'):
    # cax = ax.imshow(tf.transpose(logmel), aspect='auto', origin='lower', cmap='viridis')
    cax = ax.imshow(logmel, aspect='auto', origin='lower', cmap='viridis')
    fig.colorbar(cax, ax=ax, label='Magnitude')
    ax.set_title(title)
    ax.set_ylabel('Frequency bin')
    ax.set_xlabel('Time frame')


if __name__ == "__main__":
    import logging
    import numpy as np

    from config import default_model, default_lang
    from whisper_s4y import _artefact_dir, _artefacts_dir, _project_root

    default_model_id = default_model.split('/')[-1]

    # Setup
    parser = argparse.ArgumentParser(description="Test package cli")

    parser.add_argument("--debug", action='store_true', help="Turn on debugging")
    parser.add_argument("--model_lang", type=str, help="The language used to recognize speech",
                        default=default_lang)
    parser.add_argument("--skip_plot", action='store_true', help="Skip the generator model")
    parser.add_argument("--model_name", type=str, help="The name of the Huggingface model",
                        default=default_model)
    parser.add_argument("--tflite_generator_path", type=str, help="The path of the Generator tflite model path",
                        default=os.path.join(_artefacts_dir, f"[model_id]-[lang].tflite"))
    parser.add_argument("--tflite_features_extractor_path", type=str, help="The path of the FeaturesExtractor tflite "
                                                                           "model path",
                        default=_artefact_dir(os.path.basename("features-extractor.tflite")))

    args = parser.parse_args()

    if args.debug:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.INFO)

    lang = args.model_lang
    huggingface_model_name = args.model_name
    huggingface_model_id = huggingface_model_name.split("/")[-1]
    tflite_generator_path = args.tflite_generator_path.replace('[model_id]',
                                                               huggingface_model_id).replace('[lang]', lang)
    tflite_features_extractor_path = args.tflite_features_extractor_path.replace('[model_id]', huggingface_model_id)

    # Functions
    tflite_generate = tflite_generator(tflite_generator_path)
    tflite_extract_features = tflite_features_extractor(tflite_features_extractor_path)

    transformers_generate = transformers_generator(huggingface_model_name, lang)
    transformers_extract_features = transformers_features_extractor(huggingface_model_name)

    decode = transformers_decoder(huggingface_model_name)

    ## 1st pass
    # Audio
    if lang == 'ar':
        audio1, title1 = audio_ar_1()
    else:
        audio1, title1 = audio_en_1()

    # LogMel
    features1t = transformers_extract_features(audio1)
    log.debug(f"transformers logmel shape: {features1t.shape}")
    features1 = tflite_extract_features(audio1)
    log.debug(f"tflite logmel shape: {features1.shape}")
    try:
        np.testing.assert_allclose(features1t.numpy(), features1.numpy(), rtol=1e-3, atol=1e-3)
    except AssertionError as e:
        log.error(f"LogMel features are not close: {e}")

    # Tokens
    tokens1t = transformers_generate(features1t)
    log.info(f"transformers tokens shape: {tokens1t.shape}")
    tokens1 = tflite_generate(features1t)
    log.info(f"tflite tokens shape: {tokens1.shape}")

    # transformersTokensInTFLiteTones = tf.map_fn(lambda x: tf.reduce_any(tf.equal(tokens1[0], x)), tokens1t, dtype=tf.bool)
    # is_subset = tf.reduce_all(transformersTokensInTFLiteTones)
    # log.info(f"Transformers tokens are subset of TFLite tokens: {is_subset}")

    transcript1t = decode(tokens1t)
    log.info(f"Transformers transcript: {transcript1t}")
    transcript1 = decode(tokens1)
    log.info(f"TFLite transcript: {transcript1}")

    if not args.skip_plot:
        fig1, axs1 = plt.subplots(2, 1, figsize=(12, 12))  # Create a figure containing a single Axes.
        fig1.canvas.manager.set_window_title(f"Audio: {title1}")
        fig1.suptitle(f"Audio: {title1}")
        plot_logmel(features1[0], fig1, axs1[0], f"LogMel (tflite)")
        plot_logmel(features1t[0], fig1, axs1[1], f"LogMel (transformers)")

        if False:
            fig2, axs2 = plt.subplots(2, 1, figsize=(12, 12))  # Create a figure containing a single Axes.
            fig2.canvas.manager.set_window_title(f"Audio: {title2}")
            fig2.suptitle(f"Audio: {title2}")
            plot_logmel(features2, fig2, axs2[0], f"LogMel (tflite)")
            plot_logmel(features2t, fig2, axs2[1], f"LogMel (transformers)")

        plt.show()
