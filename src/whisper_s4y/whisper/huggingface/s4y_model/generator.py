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
