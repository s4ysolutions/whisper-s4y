import logging
import numpy as np
import os
import tempfile
import tensorflow as tf
from typing import List
from transformers import WhisperProcessor, WhisperTokenizer, TFWhisperForConditionalGeneration, \
    TFForceTokensLogitsProcessor

log = logging.getLogger("whisper2tflite")


# Patching methods of class TFForceTokensLogitsProcessor(TFLogitsProcessor):
# TFForceTokensLogitsProcessor has a bug which causes lite model to crash
# to fix it, the 2 methods are overridden and replaced
# https://github.com/huggingface/transformers/issues/19691#issuecomment-1791869884

def patched__init__(self, force_token_map: List[List[int]]):
    force_token_map = dict(force_token_map)
    # Converts the dictionary of format {index: token} containing the tokens to be forced to an array, where the
    # index of the array corresponds to the index of the token to be forced, for XLA compatibility.
    # Indexes without forced tokens will have an negative value.
    force_token_array = np.ones((max(force_token_map.keys()) + 1), dtype=np.int32) * -1
    for index, token in force_token_map.items():
        if token is not None:
            force_token_array[index] = token
    self.force_token_array = tf.convert_to_tensor(force_token_array, dtype=tf.int32)


TFForceTokensLogitsProcessor.__init__ = patched__init__


def patched__call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int) -> tf.Tensor:
    def _force_token(generation_idx):
        batch_size = scores.shape[0]
        current_token = self.force_token_array[generation_idx]

        # Original code below generates NaN values when the model is exported to tflite
        # it just needs to be a negative number so that the forced token's value of 0 is the largest
        # so it will get chosen
        # new_scores = tf.ones_like(scores, dtype=scores.dtype) * -float("inf")
        new_scores = tf.ones_like(scores, dtype=scores.dtype) * -float(1)
        indices = tf.stack((tf.range(batch_size), tf.tile([current_token], [batch_size])), axis=1)
        updates = tf.zeros((batch_size,), dtype=scores.dtype)
        new_scores = tf.tensor_scatter_nd_update(new_scores, indices, updates)
        return new_scores

    scores = tf.cond(
        tf.greater_equal(cur_len, tf.shape(self.force_token_array)[0]),
        # If the current length is geq than the length of force_token_array, the processor does nothing.
        lambda: tf.identity(scores),
        # Otherwise, it may force a certain token.
        lambda: tf.cond(
            tf.greater_equal(self.force_token_array[cur_len], 0),
            # Only valid (positive) tokens are forced
            lambda: _force_token(cur_len),
            # Otherwise, the processor does nothing.
            lambda: scores,
        ),
    )
    return scores


TFForceTokensLogitsProcessor.__call__ = patched__call__


# A wrapper around hugging face model to be used by Lite interpretation
# will have the only function `serving` to be called by the external code
class GenerateModel(tf.Module):
    def __init__(self, model, forced_decoder_ids):
        super(GenerateModel, self).__init__()
        # actual Lite model to be used for generation
        self.model = model
        self.model.config.forced_decoder_ids = forced_decoder_ids
        self.model.generation_config.forced_decoder_ids = forced_decoder_ids
        self.forced_decoder_ids = forced_decoder_ids

    @tf.function(
        input_signature=[
            tf.TensorSpec((1, 80, 3000), tf.float32, name="input_features"),
        ],
    )
    def serving(self, input_features):
        # pass the data to the lite model to get the array of tokens
        outputs = self.model.generate(
            input_features,
            forced_decoder_ids=self.forced_decoder_ids,
            # max_new_tokens=128,  # change as needed
            return_dict_in_generate=True,
        )
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


def create_from_huggingface(model_name: str, lang: str) -> str:
    saved_model_dir = os.path.join(tempfile.gettempdir(), 'whisper2tflite', 'generator')
    log.debug("huggingface model download start...")
    model = TFWhisperForConditionalGeneration.from_pretrained(model_name)
    log.info("huggingface model download done")

    log.debug("huggingface tokenize download start...")
    processor = WhisperTokenizer.from_pretrained(model_name)
    log.info("huggingface tokenize download done")

    log.debug("generator creating start...")
    if lang is None:
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


if __name__ == "__main__":
    import argparse
    import convertor
    from config import default_model, default_lang

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

    model_path = create_from_huggingface(model_name, lang)
    convertor.convert_saved(model_path, os.path.join(artefacts_dir, generator_model_name))
