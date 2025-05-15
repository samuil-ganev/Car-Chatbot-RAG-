import sys
import logging
from pathlib import Path
from typing import List, Optional

from llama_index.core.schema import NodeWithScore
from llama_index.core import (
    VectorStoreIndex
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from data_processing import vectorizer

VECTOR_STORE_DIR = project_root / 'assets' / 'vector_store'

EMBEDDING_MODEL_PATH = str(project_root / 'assets' / 'embedding_model' / 'BAAI/bge-base-en-v1.5')
EMBEDDING_DIM = 768
EMBED_MODEL_KWARGS = {}
EMBED_DEVICE = None

SIMILARITY_TOP_K = 5
IVF_NPROBE = 16

def initialize_embed_model(model_name: str, device: Optional[str], model_kwargs: Optional[dict]) -> Optional[HuggingFaceEmbedding]:
    '''
    Initializes the HuggingFace embedding model
    '''

    logging.info(f'Initializing HuggingFace embedding model: {model_name}')
    try:

        model = HuggingFaceEmbedding(
            model_name=model_name,
            device=device,
            model_kwargs=model_kwargs
        )

        logging.info('Embedding model initialized successfully')
        return model

    except Exception as e:

        logging.error(f'Failed to initialize HuggingFace Embedding model \'{model_name}\': {e}', exc_info=True)
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
            embedding_model_name=EMBEDDING_MODEL_PATH,
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

        logging.error(f'Failed to load index: {e}', exc_info=True)
        return None
    
def search_in_index(query_str: str, index: VectorStoreIndex, top_k: int = SIMILARITY_TOP_K) -> List[NodeWithScore]:
    '''
    Performs a similiarity search on the loaded vector index using its retriever
    '''

    if not query_str:
        logging.warning('Empty query text')
        return []
    if not index:
        logging.error('Provided index object is invalid or None')
        return []
    
    logging.info(f'Performing search for query: \'{query_str}\' with top_k={top_k}')

    try:

        retriever = index.as_retriever(similarity_top_k=top_k)
        retriever_nodes = retriever.retrieve(query_str)

        if not retriever_nodes:
            logging.info('No results')
            return []
        
        logging.info(f'Search successful. Retrieved {len(retriever_nodes)} nodes')

        for i, node_with_score in enumerate(retriever_nodes):
            logging.debug(f'  Result {i+1}: Node ID: {node_with_score.node.node_id}, Score: {node_with_score.score:.4f}')

        return retriever_nodes
    
    except Exception as e:
        
        logging.error(f'An error occurred during search for query \'{query_str}\': {e}', exc_info=True)
        return []
