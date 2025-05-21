'''Search module for querying the vector index built from processed document chunks.'''
import sys
import logging
from pathlib import Path
from typing import List, Optional

from llama_index.core.schema import NodeWithScore
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from data_processing import vectorizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

VECTOR_STORE_DIR = project_root / 'assets' / 'vector_store'

EMBEDDING_MODEL_NAME = 'BAAI/bge-base-en-v1.5'
EMBEDDING_DIM = 768
EMBED_MODEL_KWARGS = {}
EMBED_DEVICE = None

SIMILARITY_TOP_K = 5
IVF_NPROBE = 16


def initialize_embed_model(model_name: str,
    device: Optional[str], model_kwargs: Optional[dict]) -> Optional[HuggingFaceEmbedding]:
    '''
    Initializes the HuggingFace embedding model.
    '''
    logging.info('Initializing HuggingFace embedding model: %s', model_name)
    try:
        model = HuggingFaceEmbedding(
            model_name=model_name,
            device=device,
            model_kwargs=model_kwargs
        )
        logging.info('Embedding model initialized successfully')
        return model
    except Exception as e:
        logging.error('Failed to initialize HuggingFace Embedding model %r: %s',
                       model_name, e, exc_info=True)
        return None

def load_vector_index(embed_model: HuggingFaceEmbedding) -> Optional[VectorStoreIndex]:
    '''
    Loads the VectorStoreIndex from the specified storage path
    '''
    if not embed_model:
        logging.error('Cannot load index without a valid embedding model')
        return None
    try:
        vector_index = vectorizer.VectorIndex(
            embedding_model_name=EMBEDDING_MODEL_NAME,
            embedding_dim=EMBEDDING_DIM,
            vector_store_path=VECTOR_STORE_DIR,
            embed_model_kwargs=EMBED_MODEL_KWARGS,
            embed_device=EMBED_DEVICE,
            ivf_nprobe=IVF_NPROBE
        )

        index = vector_index.build_or_load_index(nodes=[])

        logging.info('Successfully loaded vector index')
        return index
    except Exception as e:
        logging.error('Failed to load index: %s', e, exc_info=True)
        return None

def search_in_index(query_str: str, index: VectorStoreIndex,
                     top_k: int = SIMILARITY_TOP_K) -> List[NodeWithScore]:
    '''
    Performs a similiarity search on the loaded vector index using its retriever
    '''
    if not query_str:
        logging.warning('Empty query text')
        return []
    if not index:
        logging.error('Provided index object is invalid or None')
        return []
    logging.info('Performing search for query: %r with top_k=%d', query_str, top_k)

    try:
        retriever = index.as_retriever(similarity_top_k=top_k)
        retriever_nodes = retriever.retrieve(query_str)

        if not retriever_nodes:
            logging.info('No results')
            return []
        logging.info('Search successful. Retrieved %d nodes', len(retriever_nodes))

        for i, node_with_score in enumerate(retriever_nodes):
            logging.debug('  Result %d: Node ID: %s, Score: %.4f',
                          i+1, node_with_score.node.node_id, node_with_score.score)

        return retriever_nodes

    except Exception as e:
        logging.error('An error occurred during search for query %r: %s',
         query_str, e, exc_info=True)
        return []
