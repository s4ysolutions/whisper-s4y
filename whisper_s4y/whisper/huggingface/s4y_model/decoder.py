import tensorflow as tf

from transformers import TFLogitsProcessorList
from typing import Optional
from whisper_s4y.whisper import huggingface as hf
from whisper_s4y import tflite as _tflite, log

eos_token_id = 50257
pad_token_id = 50257
startoftranscript_token_id = 50258

input_ids = tf.constant([[startoftranscript_token_id]], dtype=tf.int32)
input_ids_seq_length = 1


class S4yDecoder(_tflite.ServingModel):
    def __init__(self, huggingface_model_id: str, lang='en', max_length: int = 56):
        if max_length > 448:
            raise ValueError("max_length must be less than or equal to 448")
        self.max_length = max_length
        fcgm = hf.for_conditional_generation(huggingface_model_id)
        tokenizer = hf.tokenizer(huggingface_model_id)
        forced_decoder_ids = tokenizer.get_decoder_prompt_ids(language=lang, task="transcribe")
        fcgm.generation_config.forced_decoder_ids = forced_decoder_ids
        self.transformer_model = fcgm

        logits_processor = TFLogitsProcessorList()
        self.logits_processor = fcgm._get_logits_processor(
            generation_config=fcgm.generation_config,
            input_ids_seq_length=input_ids_seq_length,
            logits_processor=logits_processor,
        )
        self.eos_token_id = eos_token_id
        super().__init__()

    def call(self, encoder_hidden_states: tf.TensorSpec(shape=[1, 1500, 384], dtype=tf.float32)) -> tf.TensorSpec(
        shape=[1, None], dtype=tf.int32):
        tokens = self.transformer_model.greedy_search(
            input_ids=input_ids,
            max_length=self.max_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            logits_processor=self.logits_processor,
            output_attentions=False,
            output_hidden_states=False,
            output_scores=False,
            return_dict_in_generate=False,
            use_cache=True,
            encoder_outputs=encoder_hidden_states
        )
        return tokens

    def __call__(self, encoder_hidden_states: tf.TensorSpec(shape=[1, 1500, 384], dtype=tf.float32)) -> tf.TensorSpec(
        shape=[1, None], dtype=tf.int32):
        return self.call(encoder_hidden_states)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[1, 1500, 384], dtype=tf.float32, name="encoder_hidden_states")
        ]
    )
    def serving(self, encoder_hidden_states: tf.TensorSpec(shape=[1, 1500, 384], dtype=tf.float32)):
        tokens = self(encoder_hidden_states=encoder_hidden_states)
        tensor_448 = tf.concat(
            [tokens, tf.fill((tokens.shape[0], 448 - tokens.shape[1]), eos_token_id)], axis=1)
        return {
            'tokens': tensor_448
        }

    def tflite(self, model_name: str='decoder', tflite_model_path: Optional[str] = None,
               log=log, artefact=True, optimize=True) -> str:
        return _tflite.tflite(model_name, self, tflite_model_path=tflite_model_path, log=log,
                              artefact=artefact, optimize=optimize)


if __name__ == "__main__":
    S4yDecoder('openai/whisper-tiny').tflite('s4y_decoder_optimized', artefact=False, optimize=True)
