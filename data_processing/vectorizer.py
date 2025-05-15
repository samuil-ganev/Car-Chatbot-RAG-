import logging
import faiss
import numpy as np
from pathlib import Path
from typing import List, Optional

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.schema import BaseNode
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class VectorIndex:
    '''
    Builds and loads a FAISS VectorStoreIndex
    '''

    def __init__(
        self,
        embedding_model_name: str,
        embedding_dim: int,
        vector_store_path: Path,
        embed_model_kwargs: Optional[dict] = None,
        embed_device: Optional[str] = None,
        ivf_nlist: int = 256,
        ivf_nprobe: int = 16
    ):
        self.embedding_model_name = embedding_model_name
        self.embedding_dim = embedding_dim
        self.vector_store_path = vector_store_path
        self.embed_model_kwargs = embed_model_kwargs
        self.embed_device = embed_device
        self.embed_model = self._initialize_embed_model()

        self.ivf_nlist = ivf_nlist
        self.ivf_nprobe = ivf_nprobe

    def _initialize_embed_model(self) -> Optional[HuggingFaceEmbedding]:
        '''
        Initializes the HuggingFace embedding model
        '''

        logging.info(f'Initializing HuggingFace embedding model: {self.embedding_model_name}')

        try:

            model = HuggingFaceEmbedding(
                model_name=self.embedding_model_name,
                device=self.embed_device,
                model_kwargs=self.embed_model_kwargs
            )

            # Verify dimensions:
            test_emb = model.get_text_embedding('test')
            detected_dim = len(test_emb)
            logging.info(f'Embedding model loaded. Detected dimension: {detected_dim}')

            if detected_dim != self.embedding_dim:
                logging.warning(f'EMBEDDING_DIM {self.embedding_dim} != detected {detected_dim}')
                self.embedding_dim = detected_dim
        
        except Exception as e:
            logging.error(f'Failed to initialize HuggingFace Embedding model \'{self.embedding_model_name}\': {e}', exc_info=True)
            return None
        
        return model
        
    def _load_index(self) -> Optional[VectorStoreIndex]:
        '''
        Attempts to load an existing index
        '''

        if not self.embed_model:
            logging.error('Cannot load index without valid embedding model')
            return None
        
        logging.info('Attempting to load existing vector index')

        try:

            faiss_index_path = self.vector_store_path / 'faiss.index'
            if not faiss_index_path.exists():
                raise FileNotFoundError(f'FAISS index file not found: {faiss_index_path}')
        
            faiss_index = faiss.read_index(str(faiss_index_path))
            vector_store = FaissVectorStore(faiss_index=faiss_index)

            storage_context = StorageContext.from_defaults(
                persist_dir=str(self.vector_store_path),
                vector_store=vector_store
            )
            index = load_index_from_storage(
                storage_context=storage_context,
                embed_model=self.embed_model
            )
            logging.info('Successfully loaded existing index')

            return index

        except Exception as e:

            logging.error(f'Failed to load index from {self.vector_store_path}: {e}', exc_info=True)
            return None
        
    def _create_index(self, nodes: List[BaseNode]) -> Optional[VectorStoreIndex]:
        '''
        Creates a new vector index from the provided chunks
        '''

        if not self.embed_model:
            logging.error('Cannot load index without valid embedding model')
            return None
        
        if not nodes:
            logging.error('Cannot create index: No chunks provided')
            return None
        
        logging.info(f'Creating new vector index at {self.vector_store_path} from {len(nodes)} chunks')
        self.vector_store_path.mkdir(parents=True, exist_ok=True)

        try:

            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            faiss_index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, self.ivf_nlist, faiss.METRIC_L2)
            faiss_index.nprobe = self.ivf_nprobe
            logging.info(f'Initialized FAISS index (IndexIVFFlat) with dimension {self.embedding_dim}')

            texts = [node.get_content() for node in nodes]
            embeddings = np.array(self.embed_model.get_text_embedding_batch(texts, show_progress=True), dtype='float32')

            if not faiss_index.is_trained:
                logging.info(f'Training FAISS index (IndexIVFFlat) on {len(embeddings)} vectors')
                faiss_index.train(embeddings)
                logging.info('FAISS index training complete')
            else:
                logging.info('FAISS index already trained')

            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            logging.info('Created vector database')

            index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
                embed_model=self.embed_model,
                show_progress=True
            )
            logging.info('Index creation complete')
            index.storage_context.persist(persist_dir=str(self.vector_store_path))
            faiss.write_index(faiss_index, str(self.vector_store_path / 'faiss.index'))

            return index

        except Exception as e:

            logging.error(f'Failed during index creation: {e}', exc_info=True)
            return None
        
    def build_or_load_index(self, nodes: List[BaseNode], force_reindex: bool = False) -> Optional[VectorStoreIndex]:
        '''
        Loads the index if it exists, otherwise creates a new index from the provided nodes
        '''

        if not self.embed_model:
            logging.error('Cannot load index without valid embedding model')
            return None
        
        index = None
        if not force_reindex and self.vector_store_path.exists():
            index = self._load_index()
        
        if index is None:
            index = self._create_index(nodes)
        
        return index
