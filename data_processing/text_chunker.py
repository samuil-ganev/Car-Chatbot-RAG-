import logging
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any

from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.schema import Document, TextNode
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)

MIN_CHUNK_SIZE = 250
MAX_CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150

class TextChunker:
    '''
    Chunks Markdown text into structured nodes using MarkdownNodeParser,
    then applies LangChain's RecursiveCharacterTextSplitter to enforce size bounds.
    '''

    def __init__(
        self,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        min_chunk_size_chars: int = MIN_CHUNK_SIZE,
        max_chunk_size_chars: int = MAX_CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.node_parser = MarkdownNodeParser(
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size_chars,
            chunk_overlap=chunk_overlap,
            separators=['\n\n', '\n', '. ', '? ', '! ', '... ', ' ']
        )

        self.min_chunk_size = min_chunk_size_chars
        logging.info('TextChunker initialized with recursive splitter')

    def _extract_model_name(self, text: str, filename: str) -> str:
        '''Detect car model from text or filename. Defaults to 'Volkswagen'.'''

        known_models = ['Ford Mustang', 'Daewoo Matiz', 'Honda', 'Subaru', 'Ford', 'Volkswagen']
        text_norm = text.lower()
        fn_norm = filename.lower().replace('-', ' ').replace('_', ' ')
        for model in known_models:
            if model.lower() in text_norm or model.lower() in fn_norm:
                return model
        return 'Volkswagen'

    def chunk_markdown_file(self, md_path: Path) -> Optional[List[TextNode]]:
        '''Split a Markdown file into size-bounded TextNode chunks.'''

        if not md_path.is_file() or md_path.suffix.lower() != '.md':
            logging.warning('Skipping non-markdown file: %s', md_path)
            return None

        try:
            logging.info('Chunking file: %s', md_path.name)
            content = md_path.read_text(encoding='utf-8', errors='ignore')
            model_name = self._extract_model_name(content, md_path.name)

            doc = Document(text=content, metadata={'file_name': md_path.name})
            nodes = self.node_parser.get_nodes_from_documents([doc])

            merged_buffers = []  # list of tuples (text, metadata)
            buffer_text = ''
            buffer_meta: Dict[str, Any] = {}

            for node in nodes:
                node_text = node.get_text().strip()
                if not buffer_text:
                    buffer_text = node_text
                    buffer_meta = node.metadata.copy()
                else:
                    if len(buffer_text) < self.min_chunk_size:
                        buffer_text += '\n\n' + node_text
                    else:
                        merged_buffers.append((buffer_text, buffer_meta))
                        buffer_text = node_text
                        buffer_meta = node.metadata.copy()

            if buffer_text:
                merged_buffers.append((buffer_text, buffer_meta))

            final_nodes: List[TextNode] = []
            for buf_text, meta in merged_buffers:
                full_text = f'Car model: {model_name}\n\n{buf_text}'
                chunks = self.splitter.split_text(full_text)

                for chunk in chunks:
                    node = TextNode(
                        id_=str(uuid.uuid4()),
                        text=chunk,
                        metadata={**meta, 'car_model': model_name},
                        start_char_idx=None,
                        end_char_idx=None,
                        excluded_embed_metadata_keys=[],
                        excluded_llm_metadata_keys=[]
                    )
                    final_nodes.append(node)

            logging.info('Produced %d chunks for %s', len(final_nodes), md_path.name)
            return final_nodes

        except Exception as e:
            logging.error('Error processing %s: %s', md_path.name, e, exc_info=True)
            return None

    def chunk_dir(self, dir: Path) -> Dict[str, List[TextNode]]:
        '''Process all Markdown files in a directory.'''

        if not dir.is_dir():
            logging.error('Directory not found: %s', dir)
            return {}

        all_chunks = {}
        for md_file in dir.glob('*.md'):
            chunks = self.chunk_markdown_file(md_file)
            if chunks:
                all_chunks[md_file.name] = chunks

        total = sum(len(c) for c in all_chunks.values())
        logging.info('Total chunks across directory: %d', total)
        return all_chunks
