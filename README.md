#h2 A set of python scripts to convert [TF variant of openai/whisper](https://huggingface.co/openai/whisper-base) ASR ML model to TF lite

This [pipeline](https://github.com/s4ysolutions/whishper2tflite/blob/main/.github/workflows/convert.yml) creates an artifact
with the tflite model, whisper features extractor and the assets necessary for the converting
the tokenized model output to the readable text.

Python's tensorflow package is fixed to 2.16 in requirements.txt in order to correspond to the 
org.tensorflow:tensorflow-lite maven artifact, which is the most recent in the moment
of writing.

#h2 Usage

### 1. Setup the environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the conversion huggingface model to tflite

creates the artefacts in the `./artefacts` directory with default model name and language name
```bash
python -m whisper2tflite
```

specify the model name and language name
```bash
python -m whisper2tflite --model_name="openai/whisper-base" --lang_name=ar --output_dir="artefacts"
```

### 3. Run the tests against the converted model

try to recognize the en/ar speech from the `whisper_tflite/test_data` directory
and do not show MelLogs plots

```bash
python whisper_tflite/test.py --skip_plot True 
```
