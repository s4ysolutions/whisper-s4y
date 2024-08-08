import argparse
from whisper_tflite import assets, generator, features_extractor, convertor

parser = argparse.ArgumentParser(description="A script that crates a TFLite model of Whisper and features extractor")

parser.add_argument("--model_name", type=str, help="The name of the Huggingface model", default="openai/whisper-tiny")
parser.add_argument("--skip_generator", type=str, help="Skip the generator model", default=False)
parser.add_argument("--skip_features_extractor", type=str, help="Skip the features extractor", default=False)
parser.add_argument("--skip_assets", type=str, help="Skip the downloading of assets", default=False)
parser.add_argument("--artefacts_dir", type=str, help="The directory to save the model and assets", default="artefacts")

# Parse the arguments
args = parser.parse_args()

print(args.model_name)

exit(0)
model_name = 'openai/whisper-tiny'
artefacts_dir = "artefacts"
generator_model_name = "whisper-tiny"
features_extractor_model_name = "features-extractor"

model_path = generator.create_from_huggingface(model_name)
convertor.convert_saved(model_path, "artefacts/whisper-tiny.tflite")

features_path = features_extractor.create_features_extractor()
convertor.convert_saved(features_path, "artefacts/features-extractor.tflite")

assets.download_from_huggingface(model_name, artefacts_dir)
