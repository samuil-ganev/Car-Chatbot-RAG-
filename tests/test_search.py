'''Unit tests for embedding model initialization, vector index loading,
and search functionality.'''
import sys
from pathlib import Path
#import pytest

from rag.search import initialize_embed_model, load_vector_index, search_in_index


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_embed_model_initializes():
    '''Tests that the embedding model initializes correctly.'''
    model = initialize_embed_model('BAAI/bge-base-en-v1.5', None, {})
    assert model is not None

def test_load_vector_index_returns_index():
    '''Tests that the vector index loads correctly using the embedding model.'''
    embed_model = initialize_embed_model('BAAI/bge-base-en-v1.5', None, {})
    index = load_vector_index(embed_model)
    assert index is not None

def test_search_returns_results():
    '''Tests that the search function returns a list of results.'''
    embed_model = initialize_embed_model('BAAI/bge-base-en-v1.5', None, {})
    index = load_vector_index(embed_model)
    results = search_in_index('why there is a problem with the engine of my honda?', index)
    assert isinstance(results, list)
