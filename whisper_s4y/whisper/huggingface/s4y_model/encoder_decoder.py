import tensorflow as tf

from whisper_s4y.whisper.huggingface.s4y_model import S4yEncoder, S4yDecoder
from whisper_s4y import tflite as _tflite, log


class S4yEncoderDecoder(_tflite.ServingModel):
    def __init__(self, huggingface_model_id: str, lang: str, max_length: int = 112):
        self.encoder = S4yEncoder(huggingface_model_id)
        self.decoder = S4yDecoder(huggingface_model_id, lang=lang, max_length=max_length)
        super().__init__()

    def call(self, input_features):
        encoded_output = self.encoder(input_features)
        return self.decoder(encoded_output)

    def __call__(self, input_features):
        return self.call(input_features)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[1, 80, 3000], dtype=tf.float32, name="input_features")
        ]
    )
    def serving(self, input_features: tf.TensorSpec(shape=[1, 80, 3000], dtype=tf.float32)):
        tokens = self(input_features=input_features)
        tensor_448 = tf.concat(
            [tokens, tf.fill((tokens.shape[0], 448 - tokens.shape[1]), self.decoder.eos_token_id)], axis=1)
        return {
            'tokens': tensor_448
        }

    def tflite(self, model_name='encoder_decoder', tflite_model_path=None,
               log=log, optimize=False, artefact=True) -> str:
        return _tflite.tflite(model_name, self, tflite_model_path, log=log, optimize=optimize, artefact=artefact)


if __name__ == "__main__":
    from whisper_s4y import log
    S4yEncoderDecoder('openai/whisper-tiny', lang='ar').tflite(log=log)
