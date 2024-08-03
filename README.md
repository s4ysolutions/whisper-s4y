#h2 A set of python scripts to convert [TF variant of openai/whisper](https://huggingface.co/openai/whisper-base) ASR ML model to TF lite

This [pipeline](https://github.com/s4ysolutions/whishper2tflite/blob/main/.github/workflows/convert.yml) creates an artifact
with the tflite models and the assets necessry for the converting the tokenazid model output to the readbale text.

Python's tensorflow package is fixed to 2.16 in requirements.txt in order to correspond to the 
org.tensorflow:tensorflow-lite maven artifact, which is the most recent in the moment
of writing.
