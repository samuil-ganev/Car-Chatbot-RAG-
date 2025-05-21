import pickle
import json
import logging
from typing import List
from pathlib import Path

from llama_index.core.schema import TextNode
from docling_converter import PDFConverter
from describe import MarkdownImageProcessor
from text_chunker import TextChunker
from vectorizer import VectorIndex

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main(modes: List[str] = []):
    '''
    1. Convert PDFs to MDs
    2. Chunk MD files
    3. Build/load vector index
    '''
    project_root = Path(__file__).parent.parent
    RAW_PDF_DIR = project_root / 'assets' / 'pdfs'
    MARKDOWN_OUTPUT_DIR = project_root / 'assets' / 'markdown'
    VECTOR_STORE_DIR = project_root / 'assets' / 'vector_store'
    IRPA_JSON = project_root / 'assets' / 'secterts' / 'credentials.json'

    EMBEDDING_MODEL_NAME = 'BAAI/bge-base-en-v1.5'
    EMBEDDING_DIM = 768
    EMBED_MODEL_KWARGS = {}
    EMBED_DEVICE = None

    MIN_CHUNK_TARGET_SIZE = 250

    FORCE_REPARSE_PDF = False
    FORCE_RECHUNK = False
    FORCE_REINDEX = False

    with open(IRPA_JSON, 'r') as f:
        creds = json.load(f)

    hub_base_url = creds.get('serviceurls', {}).get('AI_API_URL')
    hub_auth_url = creds.get('url')
    hub_client_id = creds.get('clientid')
    hub_client_secret = creds.get('clientsecret')

    if not RAW_PDF_DIR.exists() and not MARKDOWN_OUTPUT_DIR.exists():
        logging.error(
            'Neither PDF input dir (%s) nor MD dir (%s) exist',
            RAW_PDF_DIR, MARKDOWN_OUTPUT_DIR
        )
        print('Error: Need either raw PDFs or existing MDs')
        return

    all_nodes = []

    # --- Step 1: PDF Conversion ---
    if 'convert' in modes or not modes:
        needs_conversion = (
            FORCE_REPARSE_PDF
            or not MARKDOWN_OUTPUT_DIR.exists()
            or not any(MARKDOWN_OUTPUT_DIR.glob('*.md'))
        )
        markdown_files_generated_or_found = []

        if needs_conversion:
            logging.info('--- Starting PDF to MD Conversion ---')

            if not RAW_PDF_DIR.exists() or not any(RAW_PDF_DIR.glob('*.pdf')):
                logging.warning('PDF directory %s is empty or missing', RAW_PDF_DIR)

                if MARKDOWN_OUTPUT_DIR.exists() and any(
                    MARKDOWN_OUTPUT_DIR.glob('*.md')
                ):
                    print('Skipping conversion, using existing Markdown')
                    needs_conversion = False
                    markdown_files_generated_or_found = list(
                        MARKDOWN_OUTPUT_DIR.glob('*.md')
                    )
                else:
                    print('Error: No PDFs or existing MDs found')
                    return
            else:
                try:
                    # Uncomment to activate
                    # converter = PDFConverter()
                    # mds_generated = converter.process_dir(
                    #     RAW_PDF_DIR, MARKDOWN_OUTPUT_DIR
                    # )
                    #
                    # processor = MarkdownImageProcessor(
                    #     hub_base_url=hub_base_url,
                    #     hub_auth_url=hub_auth_url,
                    #     hub_client_id=hub_client_id,
                    #     hub_client_secret=hub_client_secret,
                    #     context_words=150
                    # )
                    # processor.process_dir(MARKDOWN_OUTPUT_DIR)

                    logging.info(
                        'Docling conversion finished. Found %d MDs',
                        len(markdown_files_generated_or_found)
                    )

                    if not markdown_files_generated_or_found and not any(
                        MARKDOWN_OUTPUT_DIR.glob('*.md')
                    ):
                        logging.error(
                            'Docling conversion produced no output and no MDs found'
                        )
                        return
                except Exception as e:
                    logging.error(
                        'Error during Docling conversion: %s', e, exc_info=True
                    )
                    return
        else:
            logging.info(
                '--- Skipping PDF Conversion (using existing Markdown) ---'
            )
            markdown_files_generated_or_found = list(
                MARKDOWN_OUTPUT_DIR.glob('*.md')
            )

    # --- Step 2: Text Chunking ---
    if 'chunk' in modes or not modes:
        mds_generated = list(MARKDOWN_OUTPUT_DIR.glob('*.md'))

        if not mds_generated:
            logging.warning('No MD files found. Skipping chunking.')
        else:
            print('\n--- Starting Markdown Text Chunking ---')
            chunker = TextChunker(min_chunk_size_chars=MIN_CHUNK_TARGET_SIZE)
            chunked_data = chunker.chunk_dir(MARKDOWN_OUTPUT_DIR)

            if chunked_data:
                total_chunks = 0
                for _, nodes in chunked_data.items():
                    all_nodes.extend(nodes)
                    total_chunks += len(nodes)

                logging.info(
                    'Chunking complete. %d chunks from %d files',
                    total_chunks, len(chunked_data)
                )
            else:
                logging.warning(
                    'Chunking resulted in zero nodes, though Markdown files exist.'
                )

        if all_nodes:
            chunks_path1 = project_root / 'assets' / 'chunks' / 'chunks.pkl'
            chunks_path2 = project_root / 'assets' / 'chunks' / 'chunks.json'

            logging.info('Saving %d chunks to %s', len(all_nodes), chunks_path1)

            try:
                with open(chunks_path1, 'wb') as f:
                    pickle.dump(all_nodes, f)
                print('Successfully saved chunks (pickle)')

                serializable_chunks = [node.to_dict() for node in all_nodes]
                with open(chunks_path2, 'w', encoding='utf-8') as f:
                    json.dump(serializable_chunks, f, indent=2)
                print('Successfully saved chunks (json)')

            except Exception as e:
                logging.error('Failed to save chunks: %s', e, exc_info=True)

    # --- Step 3: Embedding & Indexing ---
    if 'vectorize' in modes or not modes:
        if not all_nodes:
            chunks_json = project_root / 'assets' / 'chunks' / 'chunks.json'
            with open(chunks_json, 'r', encoding='utf-8') as f:
                nodes_data = json.load(f)
                for node_dict in nodes_data:
                    node = TextNode.model_validate(node_dict)
                    all_nodes.append(node)

        vector_index = VectorIndex(
            embedding_model_name=EMBEDDING_MODEL_NAME,
            embedding_dim=EMBEDDING_DIM,
            vector_store_path=VECTOR_STORE_DIR,
            embed_model_kwargs=EMBED_MODEL_KWARGS,
            embed_device=EMBED_DEVICE
        )

        index = vector_index.build_or_load_index(
            nodes=all_nodes,
            force_reindex=FORCE_REINDEX
        )

    logging.info('--- Pipeline Finished (Conversion, Chunking, Indexing) ---')


if __name__ == '__main__':
    modes = input("Enter modes ('convert', 'chunk', 'vectorize') separated by spaces: ")
    main(modes=modes.split())
