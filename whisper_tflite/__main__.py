import argparse
import logging
import os
from . import assets, generator, features_extractor, convertor

logging.basicConfig(level=logging.DEBUG)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script created TFLite version of Huggingface model,  Whisper features extractor and downloads Whisper tokenizer assets")

    parser.add_argument("--model_name", type=str, help="The name of the Huggingface model", default="openai/whisper-tiny")
    parser.add_argument("--skip_generator", type=str, help="Skip the generator model", default=False)
    parser.add_argument("--skip_features_extractor", type=str, help="Skip the features extractor", default=False)
    parser.add_argument("--skip_assets", type=str, help="Skip the downloading of assets", default=False)
    parser.add_argument("--artefacts_dir", type=str, help="The directory to save the model and assets", default="artefacts")

    # Parse the arguments
    args = parser.parse_args()

    model_name = args.model_name
    artefacts_dir = args.artefacts_dir

    generator_model_name = os.path.basename(model_name) + ".tflite"
    features_extractor_model_name = "features-extractor.tflite"

    if not args.skip_generator:
        model_path = generator.create_from_huggingface(model_name)
        convertor.convert_saved(model_path, os.path.join(artefacts_dir, generator_model_name))

    if not args.skip_features_extractor:
        features_path = features_extractor.create_features_extractor()
        convertor.convert_saved(features_path, os.path.join(artefacts_dir, features_extractor_model_name))

    if not args.skip_assets:
        assets.download_from_huggingface(model_name, artefacts_dir)
