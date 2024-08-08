import logging
import os.path
import shutil
from transformers import WhisperTokenizer
from transformers.utils import cached_file

log = logging.getLogger("whisper2tfilte")


def download_from_huggingface(model_name: str, saved_model_dir: str):
    tokenizer = WhisperTokenizer.from_pretrained(model_name)
    for asset in [cached_file(model_name, f) for f in tokenizer.vocab_files_names.values()]:
        print(os.path.basename(asset))
        shutil.copy(asset, os.path.join(saved_model_dir, os.path.basename(asset)))
