import pytest
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

EMBEDDING_MODEL_PATH = str(project_root / 'assets' / 'embedding_model' / 'BAAI/bge-base-en-v1.5')

from rag.search import initialize_embed_model, load_vector_index, search_in_index

def test_embed_model_initializes():
    model = initialize_embed_model(EMBEDDING_MODEL_PATH, None, {})
    assert model is not None

def test_load_vector_index_returns_index():
    embed_model = initialize_embed_model(EMBEDDING_MODEL_PATH, None, {})
    index = load_vector_index(embed_model)
    assert index is not None

def test_search_returns_results():
    embed_model = initialize_embed_model(EMBEDDING_MODEL_PATH, None, {})
    index = load_vector_index(embed_model)
    results = search_in_index('why there is a problem with the engine of my honda?', index)
    assert isinstance(results, list)
