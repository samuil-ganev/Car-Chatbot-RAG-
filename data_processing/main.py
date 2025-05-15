import pickle
import json

import os
import logging
from typing import List
from pathlib import Path
from dotenv import load_dotenv

from llama_index.core.schema import TextNode

from docling_converter import PDFConverter
from describe import MarkdownImageProcessor
from text_chunker import TextChunker
from vectorizer import VectorIndex

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)

def main(modes: List[str] = []):
    '''
    1. Convert PDFs to MDs
    2. Chunks MD files
    3. Builds/Loads vector index
    '''
    
    project_root = Path(__file__).parent.parent
    RAW_PDF_DIR = project_root / 'assets' / 'pdfs'
    MARKDOWN_OUTPUT_DIR = project_root / 'assets' / 'markdown'
    VECTOR_STORE_DIR = project_root / 'assets' / 'vector_store'
    IRPA_JSON = project_root / 'assets' / 'secterts' / 'credentials.json'

    EMBEDDING_MODEL_PATH = str(project_root / 'assets' / 'embedding_model' / 'BAAI/bge-base-en-v1.5')
    EMBEDDING_DIM = 768
    EMBED_MODEL_KWARGS = {}
    EMBED_DEVICE = None

    MIN_CHUNK_TARGET_SIZE = 250

    FORCE_REPARSE_PDF = False
    FORCE_RECHUNK = False
    FORCE_REINDEX = False

    with open(IRPA_JSON, 'r') as f:
        creds = json.load(f)
    hub_base_url = creds.get('serviceurls').get('AI_API_URL')
    hub_auth_url = creds.get('url')
    hub_client_id = creds.get('clientid')
    hub_client_secret = creds.get('clientsecret')

    if not RAW_PDF_DIR.exists() and not MARKDOWN_OUTPUT_DIR.exists():
        logging.error(f'Neither PDF input dir ({RAW_PDF_DIR}) nor MD dir ({MARKDOWN_OUTPUT_DIR}) exist')
        print('Error: Need either raw PDFs or existing MDs')
        return
    
    all_nodes = []

    # --- Step 1: PDF Conversion (Docling) ---

    if 'convert' in modes or modes == []:
        needs_conversion = FORCE_REPARSE_PDF or not MARKDOWN_OUTPUT_DIR.exists() or not any(MARKDOWN_OUTPUT_DIR.glob('*.md'))
        markdown_files_generated_or_found = []

        if needs_conversion:
            logging.info('--- Starting PDF to MD Conversion ---')
            if not RAW_PDF_DIR.exists() or not any(RAW_PDF_DIR.glob('*.pdf')):
                logging.warning(f'PDF directory {RAW_PDF_DIR} is empty or missing')
                
                if MARKDOWN_OUTPUT_DIR.exists() and any(MARKDOWN_OUTPUT_DIR.glob('*.md')):
                    print('Skipping conversion attempt, proceeding with existing Markdown')
                    needs_conversion = False
                    markdown_files_generated_or_found = list(MARKDOWN_OUTPUT_DIR.glob('*.md'))
                else:
                    print('Error: No PDFs to convert and no existing MDs found')
                    return
            else:
                try:
                    
                    # converter = PDFConverter()
                    # mds_generated = converter.process_dir(RAW_PDF_DIR, MARKDOWN_OUTPUT_DIR)
                    
                    # processor = MarkdownImageProcessor(
                    #     hub_base_url=hub_base_url,
                    #     hub_auth_url=hub_auth_url,
                    #     hub_client_id=hub_client_id,
                    #     hub_client_secret=hub_client_secret,
                    #     context_words=150
                    # )
                    # processor.process_dir(MARKDOWN_OUTPUT_DIR)

                    logging.info(f'Docling conversion finished. Found {len(mds_generated)} MDs')
                    
                    if not markdown_files_generated_or_found and not any(MARKDOWN_OUTPUT_DIR.glob('*.md')):
                        logging.error('Docling conversion did not produce files, and no MD files found')
                        return
                
                except Exception as e:
                
                    logging.error(f'Error during Docling conversion process: {e}', exc_info=True)
                    return
        
        else:
            logging.info('--- Skipping PDF Conversion (using existing Markdown) ---')
            mds_generated = list(MARKDOWN_OUTPUT_DIR.glob('*.md'))


    # --- Step 2: Text Chunking (LlamaIndex) ---
    
    if 'chunk' in modes or modes == []:
    
        mds_generated = list(MARKDOWN_OUTPUT_DIR.glob('*.md'))

        if not mds_generated:
            logging.warning('No MD files found. Skipping chunking')
        else:
            print('\n--- Starting Markdown Text Chunking ---')

            chunker = TextChunker(min_chunk_size_chars=MIN_CHUNK_TARGET_SIZE)
            chunked_data = chunker.chunk_dir(MARKDOWN_OUTPUT_DIR)
            
            if chunked_data:
                total_chunks = 0
                for filename, nodes in chunked_data.items():
                    # md_file_path = MARKDOWN_OUTPUT_DIR / filename
                    # for node in nodes:
                    #     node.metadata['file_path'] = str(md_file_path)
                    all_nodes.extend(nodes)
                    total_chunks += len(nodes)
                
                logging.info(f'Chunking and merging complete. Generated {total_chunks} chunks from {len(chunked_data)} files')

            else:
                logging.warning('Chunking process resulted in zero nodes, though Markdown files exist')

        if all_nodes:
            chunks_output_path1 = project_root / 'assets' / 'chunks' / 'chunks.pkl'
            chunks_output_path2 = project_root / 'assets' / 'chunks' / 'chunks.json'
            logging.info(f'Saving {len(all_nodes)} chunks to {chunks_output_path1}')

            try:

                with open(chunks_output_path1, 'wb') as f:
                    pickle.dump(all_nodes, f)
                print(f'Successfully saved chunks (pickle)')

                serializable_chunks = [node.to_dict() for node in all_nodes]
                with open(chunks_output_path2, 'w', encoding='utf-8') as f:
                    json.dump(serializable_chunks, f, indent=2)
                print(f'Successfully saved chunks (json)')

            except Exception as e:

                logging.error(f'Failed to save chunks: {e}', exc_info=True)

    # --- Step 3: Embedding and Indexing (VectorIndex) ---

    if 'vectorize' in modes or modes == []:

        if not all_nodes:
            CHUNKS_JSON_PATH = project_root / 'assets' / 'chunks' / 'chunks.json'

            with open(CHUNKS_JSON_PATH, 'r', encoding='utf-8') as f:
                nodes_data = json.load(f)

                for node_dict in nodes_data:
                    node = TextNode.model_validate(node_dict)
                    all_nodes.append(node)
        
        vector_index = VectorIndex(
            embedding_model_name=EMBEDDING_MODEL_PATH,
            embedding_dim=EMBEDDING_DIM,
            vector_store_path=VECTOR_STORE_DIR,
            embed_model_kwargs=EMBED_MODEL_KWARGS,
            embed_device=EMBED_DEVICE
        )

        index = vector_index.build_or_load_index(
            nodes=all_nodes,
            force_reindex=FORCE_REINDEX
        )

    logging.info('--- Data Preparation Pipeline Finished (Conversion, Chunking & Indexing) ---')


if __name__ == '__main__':
    modes = input('Enter modes (\'convert\', \'chunk\', \'vectorize\') separated by spaces: ')
    main(modes=modes.split())