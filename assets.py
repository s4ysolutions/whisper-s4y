import logging
import os.path
import shutil
from settings import model_name, tflite_model_path
from transformers import WhisperProcessor, WhisperTokenizer
from transformers.utils import cached_file

logging.basicConfig(level=logging.DEBUG)

processor = WhisperProcessor.from_pretrained(model_name)
tokenizer = processor.tokenizer

for asset in [cached_file(model_name, f) for f in tokenizer.vocab_files_names.values()]:
    print(os.path.basename(asset))
    shutil.copy(asset, os.path.basename(asset))
