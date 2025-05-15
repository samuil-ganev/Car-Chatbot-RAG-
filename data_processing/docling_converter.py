import logging
from pathlib import Path
from typing import Dict, Optional, List, Any

from docling.datamodel.base_models import InputFormat
from docling_core.types.doc import ImageRefMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    EasyOcrOptions
)

DOCLING_AVAILABLE = True

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PDFConverter:
    '''
    Converts PDFs to MD files
    '''
    
    def __init__(self, pdf_options: Optional[Dict] = None):
        self.converter_options = pdf_options or self._get_default_pdf_options()
        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=self.converter_options)
            }
        )
        logging.info('Docling DocumentConverter initialized')

    def _get_default_pdf_options(self) -> PdfPipelineOptions:
        '''
        Provides default configuration enabling OCR, tables and images
        '''
        logging.info('Using default Docling PDF options (OCR, Tables, Iamge enabled)')
        return PdfPipelineOptions(
            do_table_structure=True,
            do_ocr=False,
            ocr_options=EasyOcrOptions(force_full_page_ocr=True, lang=['en']),
            table_structure_options=dict(
                do_cell_matching=False,
                mode=TableFormerMode.ACCURATE
            ),
            generate_page_images=True,
            generate_picture_images=True
        )
    
    def convert_pdf_to_md(self, pdf_path: Path, output_dir: Path) -> Optional[Path]:
        '''
        Converts a single PDF to a MD file, saving referenced images
        '''
        if not pdf_path.is_file() or pdf_path.suffix.lower() != '.pdf':
            logging.warning(f'Skipping non-PDF file: {pdf_path}')
            return None
        
        md_filename = pdf_path.stem + '.md'
        outpud_md_path = output_dir / md_filename

        if outpud_md_path.exists():
            logging.info(f'MD file already exists, skipping conversion: {outpud_md_path.name}')
            return outpud_md_path
        
        logging.info(f'Starting Docling conversion for {pdf_path.name}')
        try:
            
            result = self.doc_converter.convert(str(pdf_path))
            if not result or not result.document:
                logging.error(f'Docling conversion returned empty result for {pdf_path.name}')
                return None
            
            output_dir.mkdir(parents=True, exist_ok=True)

            result.document.save_as_markdown(
                outpud_md_path,
                image_mode=ImageRefMode.REFERENCED
            )

            logging.info(f'Successfully saved MD to: {outpud_md_path}')

            return outpud_md_path
        
        except Exception as e:
            
            logging.error(f'Error converting {pdf_path.name} with Docling: {e}', exc_info=True)

    def process_dir(self, input_dir: Path, output_dir: Path) -> List[Path]:
        '''
        Processes all PDFs in a directory using Docling
        '''
        if not input_dir.is_dir():
            logging.error(f'Input directory not found: {input_dir}')
            return []
        
        output_dir.mkdir(parents=True, exist_ok=True)
        processed_files = []

        logging.info(f'Starting Docling PDF processing in directory {input_dir}')
        pdf_files = list(input_dir.glob('*.pdf'))

        if not pdf_files:
            logging.warning(f'No PDFs found in {input_dir}')
            return []
        logging.info(f'Found {len(pdf_files)} PDFs')

        for pdf_file in pdf_files:
            output_path = self.convert_pdf_to_md(pdf_file, output_dir)
            if output_path:
                processed_files.append(output_path)
            logging.info(f'Finished Docling processing. Converted {len(processed_files)} PDFs')
        
        return processed_files
