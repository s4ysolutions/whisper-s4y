from huggingface_hub import hf_hub_download
from settings import model_name

# Download the file
vocab_file = hf_hub_download(repo_id=model_name, filename="vocab.json")

# Print the path to the downloaded file
print(f"Downloaded vocab.json to: {vocab_file}")

