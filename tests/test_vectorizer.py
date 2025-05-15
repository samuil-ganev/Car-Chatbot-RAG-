import pytest
import sys
from pathlib import Path
from llama_index.core.schema import TextNode

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_processing.vectorizer import VectorIndex

VECTOR_STORE_PATH = project_root / 'assets' / 'vector_store'
EMBEDDING_MODEL_PATH = str(project_root / 'assets' / 'embedding_model' / 'BAAI/bge-base-en-v1.5')

@pytest.fixture
def test_initialize_embedding_model():
    index = VectorIndex(
        embedding_model_name=EMBEDDING_MODEL_PATH,
        embedding_dim=768,
        vector_store_path=VECTOR_STORE_PATH
    )
    assert index.embed_model is not None

def test_build_index_creates_index():
    index = VectorIndex(
        embedding_model_name=EMBEDDING_MODEL_PATH,
        embedding_dim=768,
        vector_store_path=VECTOR_STORE_PATH
    )
    vector_index = index.build_or_load_index(nodes=[], force_reindex=False)
    assert vector_index is not None
