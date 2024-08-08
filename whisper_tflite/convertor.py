import logging
import os
from typing import Union
from typing_extensions import LiteralString

import tensorflow as tf

log = logging.getLogger("whisper2tfilte")


def convert_saved(saved_model_dir: str, tflite_model_path: Union[LiteralString, str, bytes]):
    name = os.path.basename(tflite_model_path)
    log.info(f"{name} load start...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    log.info(f"{name} load done")
    log.debug(f"{name} converting start...")
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    tflite_model = converter.convert()
    log.info(f"{name} converting done")
    log.debug(f"{name} converted model save start...")
    dir = os.path.dirname(tflite_model_path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    log.info(f"{name} converted model save done")
