import logging
import os
from . import assets, generator, features_extractor, convertor

log = logging.getLogger("whisper2tflite")

if __name__ == "__main__":
    import argparse
    from config import default_model

    _cwd = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(_cwd)

    parser = argparse.ArgumentParser(
        description="The script creates TFLite version of Huggingface model,  Whisper features extractor and downloads Whisper tokenizer assets")

    parser.add_argument("--debug", type=str, help="Turn on debugging", default=True)
    parser.add_argument("--model_name", type=str, help="The name of the Huggingface model",
                        default=default_model)
    parser.add_argument("--model_lang", type=str, help="The language used to recognize speech",
                        default="en")
    parser.add_argument("--skip_generator", type=str, help="Skip the generator model", default=False)
    parser.add_argument("--skip_features_extractor", type=str, help="Skip the features extractor", default=False)
    parser.add_argument("--skip_assets", type=str, help="Skip the downloading of assets", default=False)
    parser.add_argument("--artefacts_dir", type=str, help="The directory to save the model and assets",
                        default=os.path.join(_root, "artefacts"))

    # Parse the arguments
    args = parser.parse_args()

    if args.debug:
        log.basicConfig(level=logging.DEBUG)

    model_name = args.model_name
    artefacts_dir = args.artefacts_dir

    generator_model_name = os.path.basename(model_name) + ".tflite"
    features_extractor_model_name = "features-extractor.tflite"

    if not args.skip_generator:
        model_path = generator.create_from_huggingface(model_name, args.model_lang)
        convertor.convert_saved(model_path, os.path.join(artefacts_dir, generator_model_name))

    if not args.skip_features_extractor:
        features_path = features_extractor.create_features_extractor()
        convertor.convert_saved(features_path, os.path.join(artefacts_dir, features_extractor_model_name))

    if not args.skip_assets:
        assets.download_from_huggingface(model_name, artefacts_dir)
