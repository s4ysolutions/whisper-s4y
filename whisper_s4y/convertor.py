import logging
import os
from typing import Union
from typing_extensions import LiteralString

import tensorflow as tf

log = logging.getLogger("whisper2tfilte")


def convert_saved(saved_model_dir: str, tflite_model_path: Union[LiteralString, str, bytes], optimize: bool = True):
    name = os.path.basename(tflite_model_path)
    log.info(f"{name} load start...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    log.info(f"{name} load done")
    log.debug(f"{name} converting start...")
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
    if optimize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    else:
        converter.optimizations = []
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Converts a saved model into a TFLite model")
    parser.add_argument("--debug", action='store_true', help="Turn on debugging")
    parser.add_argument("--saved_model_dir", type=str, help="The directory of the saved model")
    parser.add_argument("--tflite_model_path", type=str, help="The path to save the TFLite model")
    parser.add_argument("--optimize", action='store_false', help="Optimize the TFLite model")
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    convert_saved(args.saved_model_dir, args.tflite_model_path, args.optimize)