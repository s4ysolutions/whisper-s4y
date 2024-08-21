import logging
import os
from . import assets, features_extractor, convertor
from whisper_tflite.generator import huggingface

if __name__ == "__main__":
    import argparse
    from . import config

    log = logging.getLogger("whisper2tflite")

    default_model = config.default_model
    default_lang = config.default_lang

    _cwd = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(_cwd)

    parser = argparse.ArgumentParser(
        description="The script creates TFLite version of Huggingface model,  Whisper features extractor and downloads Whisper tokenizer assets")

    parser.add_argument("--debug", action='store_true', help="Turn on debugging")
    parser.add_argument("--model_name", type=str, help="The name of the Huggingface model",
                        default=default_model)
    parser.add_argument("--model_lang", type=str, help="The language used to recognize speech",
                        default=default_lang)
    parser.add_argument("--skip_generator", action='store_true', help="Skip the generator model")
    parser.add_argument("--skip_features_extractor", action='store_true', help="Skip the features extractor")
    parser.add_argument("--skip_assets", action='store_true', help="Skip the downloading of assets")
    parser.add_argument("--artefacts_dir", type=str, help="The directory to save the model and assets",
                        default=os.path.join(_root, "artefacts"))
    parser.add_argument("--saved_model_dir", type=str, help="The directory to save the intermediate model files",
                        default=None)

    # Parse the arguments
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    model_name = args.model_name
    model_lang = args.model_lang
    artefacts_dir = args.artefacts_dir
    model_id = model_name.split("/")[-1]
    generator_model_name = f"{model_id}-{model_lang}.tflite"
    features_extractor_model_name = f"features-extractor.tflite"
    saved_model_dir = args.saved_model_dir

    if not args.skip_generator:
        model_path = huggingface.save(model_name, model_lang, saved_model_dir)
        if (args.debug):
            log.debug(f"Model saved path: {model_path}")
        convertor.convert_saved(model_path, os.path.join(artefacts_dir, generator_model_name))

    if not args.skip_features_extractor:
        features_path = features_extractor.create_features_extractor()
        convertor.convert_saved(features_path, os.path.join(artefacts_dir, features_extractor_model_name), False)

    if not args.skip_assets:
        assets.download_from_huggingface(model_name, artefacts_dir)
