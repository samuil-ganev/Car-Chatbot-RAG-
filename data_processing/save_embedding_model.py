from sentence_transformers import SentenceTransformer
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

original_hf_model_name = 'BAAI/bge-base-en-v1.5'

local_model_directory = project_root / 'assets' / 'embedding_model'
local_model_path = local_model_directory / 'BAAI/bge-base-en-v1.5'

if not os.path.exists(local_model_path):
    print(f'Local model path {local_model_path} does not exist')
    print(f'Downloading {original_hf_model_name} from Hugging Face Hub...')

    os.makedirs(local_model_directory, exist_ok=True)

    model = SentenceTransformer(original_hf_model_name)

    print(f'Saving model to {local_model_path}...')
    model.save(str(local_model_path))
    print(f'Model successfully saved to {local_model_path}')
else:
    print(f'Model already exists at {local_model_path}. Skipping download and save.')
