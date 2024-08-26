import tensorflow as tf

from whisper_s4y import tflite as _tflite, log
from whisper_s4y.features_extractor import S4yFeaturesExtractor
from whisper_s4y.whisper.huggingface.s4y_model import S4yEncoder


class S4yPcmEncoder(_tflite.ServingModel):
    def __init__(self, huggingface_model_id: str):
        self.features_extractor = S4yFeaturesExtractor()
        self.transformers_model = S4yEncoder(huggingface_model_id)
        super().__init__()

    def call(self, normalized_audio):
        features = self.features_extractor(normalized_audio)
        return self.transformers_model(features)

    def __call__(self, normalized_audio):
        return self.call(normalized_audio)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=480000, dtype=tf.float32, name="normalized_audio"),
        ],
    )
    def serving(self, normalized_audio):
        output = self.call(normalized_audio)
        return {'last_hidden_state': output.last_hidden_state}

    def tflite(self, model_name='pcm-encoder', tflite_model_path=None,
               log=log, optimize=False, artefact=True) -> str:
        return _tflite.tflite(model_name, self, tflite_model_path=tflite_model_path, optimize=optimize, log=log, artefact=artefact)


if __name__ == "__main__":
    from whisper_s4y import log

    S4yPcmEncoder('openai/whisper-tiny').tflite(log=log, artefact=True)
