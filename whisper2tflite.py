import tensorflow as tf
from transformers import WhisperProcessor, TFWhisperForConditionalGeneration, TFForceTokensLogitsProcessor
import TFForceTokensLogitsProcessorPatch as patch

from settings import model_name, tflite_model_path
from test import test

# temporary catalog for saving the huggingface model localy
saved_model_dir = 'tf_whisper_saved'
# True for testing purpose
skip_convert = False
# False if test after creation
skip_test = False

# Patching methods of class TFForceTokensLogitsProcessor(TFLogitsProcessor):
# TFForceTokensLogitsProcessor has a bug which causes lite model to crach
# to fix it, the 2 methods are overriden and replaced
# https://github.com/huggingface/transformers/issues/19691#issuecomment-1791869884
TFForceTokensLogitsProcessor.__init__ = patch.patched__init__
TFForceTokensLogitsProcessor.__call__ = patch.patched__call__


# A wrapper around hugging face model to be used by Lite interpetator
# will have the only function `serving` to be called by the exernal code
class GenerateModel(tf.Module):
    def __init__(self, model, forced_decoder_ids):
        super(GenerateModel, self).__init__()
        # actual Lite model to be used for generation
        self.model = model
        # input data (alongside with audio) for every request to recognize the voice
        # language=ar task=transcribe (not translate)
        self.forced_decoder_ids = forced_decoder_ids

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
        # ...and pasess the data to the lite model to get the array of tokens
        outputs = self.model.generate(
            input_features,
            forced_decoder_ids=self.forced_decoder_ids,
            #max_new_tokens=223,  # change as needed
            return_dict_in_generate=True,
        )
        # it could return just array of tokes from `outputs["sequences"]`
        # but i was a chiken to change the borrowed code and returns
        # a dictionary with one key-value
        return {"sequences": outputs["sequences"]}


# huggingface utility to prepare audio data for input and
# decode output tokens to redable string
processor = WhisperProcessor.from_pretrained(model_name)

# this the inpute to model's `generate` method
# will be used later
forced_decoder_ids = processor.get_decoder_prompt_ids(language="ar", task="transcribe")

if not skip_convert:
    # convert huggingface Tensorflow model to Tensorflow lite

    # Original whisper model itself has `forward` method which recognize just one tocken form
    # audio data stream. Huggingface adds a wrapper around it with the method `generate`
    # to recognize the 30sec audio data
    model = TFWhisperForConditionalGeneration.from_pretrained(model_name)
    # wrap the model with our class with `serving` method
    generate_model = GenerateModel(model=model, forced_decoder_ids=forced_decoder_ids)
    # and save this (still TensorFlow) model locally (converter can convert only such saved models)
    tf.saved_model.save(generate_model, saved_model_dir, signatures={"serving_default": generate_model.serving})

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    # Magic
#    converter.target_spec.supported_ops = [
#        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
#        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
#    ]
#    converter.optimizations = [tf.lite.Optimize.DEFAULT]
#    converter.inference_input_type = tf.float32
#    converter.inference_output_type = tf.float32
    # And now we have tflite model
    tflite_model = converter.convert()

    # Save the tflite model to the file
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

# At this point we already have tflite model, and it is good idea to check it works
if not skip_test:
    test()

# it is important to note: only model will be available in the application
# which use the model but `processor` lives only in this python script, so the application
# must implement its own way to conver PCM audio data to log-mel spectrogram
# and map the tokens to characters.
# For the latter task it is good idea to preload the `vocab` from hugging face
# and either to pass it down to the application or may be better
# add the decoding to the `serving` method of the wrapper and make it to return
# the readable string. For the getting the idea see
# [create_wav2vec2.py](https://github.com/s4ysolutions/mldemo/blob/main/voice-recognition/create_wav2vec2.py)
# It has the similar wrapper class around the original PyTorch model facebook/wav2vec2
