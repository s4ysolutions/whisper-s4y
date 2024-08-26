import tensorflow as tf

from transformers.modeling_tf_outputs import TFBaseModelOutput
from whisper_s4y import tflite as _tflite, log
from whisper_s4y.whisper.huggingface import encoder


class S4yEncoder(_tflite.ServingModel):
    def __init__(self, huggingface_model_id: str):
        self.transformers_model = encoder(huggingface_model_id)
        super().__init__()

    def call(self, input_features: tf.TensorSpec(shape=[1, 80, 3000], dtype=tf.float32)) -> TFBaseModelOutput:
        encoded = self.transformers_model(input_features, return_dict=True)
        return encoded

    def __call__(self, input_features: tf.TensorSpec(shape=[1, 80, 3000], dtype=tf.float32)) -> TFBaseModelOutput:
        return self.call(input_features)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[1, 80, 3000], dtype=tf.float32, name="input_features")
        ]
    )
    def serving(self, input_features: tf.TensorSpec(shape=[1, 80, 3000], dtype=tf.float32)):
        output = self.call(input_features)
        return {'last_hidden_state': output.last_hidden_state}

    def tflite(self, model_name='encoder', tflite_model_path=None,
               log=log, optimize=False, artefact=True) -> str:
        return _tflite.tflite(model_name, self, tflite_model_path, log=log, optimize=optimize, artefact=artefact)


if __name__ == "__main__":
    from whisper_s4y import log

    S4yEncoder('openai/whisper-tiny').tflite(log=log)
