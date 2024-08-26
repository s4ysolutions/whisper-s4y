import logging
import os
import tensorflow as tf

from typing import Optional
from whisper_s4y import log, tflite as _tflite
from whisper_s4y.whisper import huggingface as hf


# A wrapper around hugging face model to be used by Lite interpretation
# will have the only function `serving` to be called by the external code
class S4yGenerator(_tflite.ServingModel):
    def __init__(self, huggingface_model_id: str, lang='en', max_length: int = 56):
        if max_length > 448:
            raise ValueError("max_length must be less than or equal to 448")
        self.max_length = max_length
        fcgm = hf.for_conditional_generation(huggingface_model_id)
        tokenizer = hf.tokenizer(huggingface_model_id)
        forced_decoder_ids = tokenizer.get_decoder_prompt_ids(language=lang, task="transcribe")
        fcgm.generation_config.forced_decoder_ids = forced_decoder_ids
        self.transformers_model = fcgm
        super().__init__()

    def call(self, input_features: tf.TensorSpec(shape=[1, 80, 3000], dtype=tf.float32)) -> (
            tf.TensorSpec(shape=[1, 5000, 384], dtype=tf.float32)):
        return self.transformers_model.generate(input_features, return_dict_in_generate=True)

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    @tf.function(
        input_signature=[
            tf.TensorSpec((1, 80, 3000), tf.float32, name="input_features"),
        ],
    )
    def serving(self, input_features):
        # pass the data to the lite model to get the array of tokens
        outputs = self(
            input_features,
            # forced_decoder_ids=self.forced_decoder_ids,
            # max_new_tokens=128,  # change as needed
            # return_dict_in_generate=True,
        )
        return {"sequences": outputs["sequences"]}

    def tflite(self, model_name='generator', tflite_model_path: Optional[str] = None,
               log=log, optimize=False, artefact=True) -> str:
        return _tflite.tflite(model_name, self, tflite_model_path=tflite_model_path, optimize=optimize, log=log,
                              artefact=artefact)


# def save(model_name: str, lang: str, saved_model_dir: str = None) -> str:
#    if saved_model_dir is None:
#        saved_model_dir = os.path.join(tempfile.gettempdir(), 'whisper2tflite', 'generator')
#    log.debug(f"{model_name} huggingface model download start...")
#    model = TFWhisperForConditionalGeneration.from_pretrained(model_name, from_pt=True)
#    log.info(f"{model_name} huggingface model download done")
#
#    log.debug(f"{model_name} huggingface tokenize download start...")
#    processor = WhisperTokenizer.from_pretrained(model_name)
#    log.info(f"{model_name} huggingface tokenize download done")
#
#    log.debug(f"{model_name}/{lang} generator creating start...")
#    if lang is None:
#        forced_decoder_ids = processor.get_decoder_prompt_ids(task="transcribe")
#    else:
#        forced_decoder_ids = processor.get_decoder_prompt_ids(language=lang, task="transcribe")
#    # wrap the model with our class with `serving` method
#    generate_model = GenerateModel(model=model, forced_decoder_ids=forced_decoder_ids)
#    log.info(f"{model_name}/{lang} generator creating done")
#
#    log.debug(f"{model_name}/{lang} generator save start...")
#    tf.saved_model.save(generate_model, saved_model_dir, signatures={"serving_default": generate_model.serving})
#    log.info(f"{model_name}/{lang} generator save done")
#    return saved_model_dir


if __name__ == "__main__":
    import argparse
    import convertor
    from whisper_s4y.config import default_model, default_lang

    _cwd = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(_cwd)

    parser = argparse.ArgumentParser(description="Converts TFWhisperForConditionalGeneration model into a TFLite model")

    parser.add_argument("--debug", type=str, help="Turn on debugging", default=True)
    parser.add_argument("--model_name", type=str, help="The name of the Huggingface model",
                        default=default_model)
    parser.add_argument("--lang", type=str, help="The language hardcoded to the model",
                        default=default_lang)
    parser.add_argument("--artefacts_dir", type=str, help="The directory to save the model and assets",
                        default=os.path.join(_root, 'artefacts'))

    # Parse the arguments
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    lang = args.lang
    model_name = args.model_name
    artefacts_dir = args.artefacts_dir
    model_id = model_name.split("/")[-1]
    generator_model_name = f"{model_id}-{lang}.tflite"

    model_path = S4yGenerator(model_name, lang).tflite(log=log, artefact=True)
    convertor.convert_saved(model_path, os.path.join(artefacts_dir, generator_model_name))
