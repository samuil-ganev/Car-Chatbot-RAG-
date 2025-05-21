'''Tests for the VectorIndex class including embedding model initialization
and index building.'''
import sys
from pathlib import Path
import pytest
from llama_index.core.schema import TextNode
from data_processing.vectorizer import VectorIndex

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

VECTOR_STORE_PATH = project_root / 'assets' / 'vector_store'

@pytest.fixture
def dummy_nodes():
    '''Provides sample TextNode data for testing index building.'''
    return [TextNode(text='Engine oil should be changed every 10,000 km.')]

def test_initialize_embedding_model():
    '''Checks that the embedding model initializes correctly in VectorIndex.'''
    index = VectorIndex(
        embedding_model_name='BAAI/bge-base-en-v1.5',
        embedding_dim=768,
        vector_store_path=VECTOR_STORE_PATH
    )
    assert index.embed_model is not None

def test_build_index_creates_index(dummy_nodes):
    '''Verifies that the vector index can be built or loaded using dummy nodes.'''
    index = VectorIndex(
        embedding_model_name='BAAI/bge-base-en-v1.5',
        embedding_dim=768,
        vector_store_path=VECTOR_STORE_PATH
    )
    vector_index = index.build_or_load_index(nodes=dummy_nodes, force_reindex=False)
    assert vector_index is not None
