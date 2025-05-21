import logging
from pathlib import Path
from typing import Dict, Optional, List

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    EasyOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode

DOCLING_AVAILABLE = True

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class PDFConverter:
    """
    Converts PDFs to Markdown (MD) files using Docling.
    """

    def __init__(self, pdf_options: Optional[Dict] = None):
        self.converter_options = pdf_options or self._get_default_pdf_options()
        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=self.converter_options)
            }
        )
        logging.info("Docling DocumentConverter initialized")

    def _get_default_pdf_options(self) -> PdfPipelineOptions:
        """
        Provides default configuration enabling OCR, tables and images.
        """
        logging.info("Using default Docling PDF options (OCR, Tables, Image enabled)")
        return PdfPipelineOptions(
            do_table_structure=True,
            do_ocr=False,
            ocr_options=EasyOcrOptions(force_full_page_ocr=True, lang=["en"]),
            table_structure_options={"do_cell_matching": False, "mode": TableFormerMode.ACCURATE},
            generate_page_images=True,
            generate_picture_images=True,
        )

    def convert_pdf_to_md(self, pdf_path: Path, output_dir: Path) -> Optional[Path]:
        """
        Converts a single PDF to a Markdown file, saving referenced images.
        """
        if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
            logging.warning("Skipping non-PDF file: %s", pdf_path)
            return None

        md_filename = pdf_path.stem + ".md"
        output_md_path = output_dir / md_filename

        if output_md_path.exists():
            logging.info("MD file already exists, skipping conversion: %s", output_md_path.name)
            return output_md_path

        logging.info("Starting Docling conversion for %s", pdf_path.name)
        try:
            result = self.doc_converter.convert(str(pdf_path))
            if not result or not result.document:
                logging.error("Docling conversion returned empty result for %s", pdf_path.name)
                return None

            output_dir.mkdir(parents=True, exist_ok=True)

            result.document.save_as_markdown(
                output_md_path,
                image_mode=ImageRefMode.REFERENCED,
            )

            logging.info("Successfully saved MD to: %s", output_md_path)

            return output_md_path

        except Exception:
            logging.exception("Error converting %s with Docling", pdf_path.name)
            return None

    def process_dir(self, input_dir: Path, output_dir: Path) -> List[Path]:
        """
        Processes all PDFs in a directory using Docling.
        """
        if not input_dir.is_dir():
            logging.error("Input directory not found: %s", input_dir)
            return []

        output_dir.mkdir(parents=True, exist_ok=True)
        processed_files = []

        logging.info("Starting Docling PDF processing in directory %s", input_dir)
        pdf_files = list(input_dir.glob("*.pdf"))

        if not pdf_files:
            logging.warning("No PDFs found in %s", input_dir)
            return []

        logging.info("Found %d PDFs", len(pdf_files))

        for pdf_file in pdf_files:
            output_path = self.convert_pdf_to_md(pdf_file, output_dir)
            if output_path:
                processed_files.append(output_path)

        logging.info("Finished Docling processing. Converted %d PDFs", len(processed_files))

        return processed_files