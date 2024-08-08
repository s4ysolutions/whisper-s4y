import logging
import os
import tempfile
import tensorflow as tf
from . import TFForceTokensLogitsProcessorPatch as patch

from transformers import WhisperProcessor, WhisperTokenizer, TFWhisperForConditionalGeneration, \
    TFForceTokensLogitsProcessor

log = logging.getLogger("whisper2tfilte")

# Patching methods of class TFForceTokensLogitsProcessor(TFLogitsProcessor):
# TFForceTokensLogitsProcessor has a bug which causes lite model to crash
# to fix it, the 2 methods are overridden and replaced
# https://github.com/huggingface/transformers/issues/19691#issuecomment-1791869884
TFForceTokensLogitsProcessor.__init__ = patch.patched__init__
TFForceTokensLogitsProcessor.__call__ = patch.patched__call__


# A wrapper around hugging face model to be used by Lite interpretation
# will have the only function `serving` to be called by the external code
class GenerateModel(tf.Module):
    def __init__(self, model, forced_decoder_ids):
        super(GenerateModel, self).__init__()
        # actual Lite model to be used for generation
        self.model = model
        self.model.config.forced_decoder_ids = forced_decoder_ids

    # signature of the only function of the cla
    @tf.function(
        # shouldn't need static batch size, but throws exception without it (needs to be fixed)
        input_signature=[
            tf.TensorSpec((1, 80, 3000), tf.float32, name="input_features"),
        ],
    )
    def serving(self, input_features):
        # it has only on input, namely tensor with audio data (mel spectrogram)
        # see the @tf.function decorator above
        # ...and passes the data to the lite model to get the array of tokens
        outputs = self.model.generate(
            input_features,
            # forced_decoder_ids=self.forced_decoder_ids,
            # max_new_tokens=223,  # change as needed
            return_dict_in_generate=True,
        )
        # it could return just array of tokes from `outputs["sequences"]`
        # but I was a chicken to change the borrowed code and returns
        # a dictionary with one key-value
        return {"sequences": outputs["sequences"]}


r"""
Constructs a Whisper Generator model for TFLite and saves it to the temporary directory.
This save model can be used to convert it to the TFLite model.

Args:
    model_name (`str`):
        The name of the wisper model as it is stated in the Hugging Face model hub.
    lang (`str`, defaults to None):
        The language code to be be used for the forced_decoder_ids
        
Returns:
    `str`: The path to the saved model directory.
"""


def create_from_huggingface(model_name: str, lang: str = None) -> str:
    saved_model_dir = os.path.join(tempfile.gettempdir(), 'whisper2tflite', 'generator')
    log.debug("huggingface model download start...")
    model = TFWhisperForConditionalGeneration.from_pretrained(model_name)
    log.info("huggingface model download done")

    log.debug("huggingface tokenize download start...")
    processor = WhisperTokenizer.from_pretrained(model_name)
    log.info("huggingface tokenize download done")

    log.debug("generator creating start...")
    if (lang is None):
        forced_decoder_ids = processor.get_decoder_prompt_ids(task="transcribe")
    else:
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=lang, task="transcribe")
    # wrap the model with our class with `serving` method
    generate_model = GenerateModel(model=model, forced_decoder_ids=forced_decoder_ids)
    log.info("generator creating done")

    log.debug("generator save start...")
    tf.saved_model.save(generate_model, saved_model_dir, signatures={"serving_default": generate_model.serving})
    log.info("generator save done")
    return saved_model_dir
