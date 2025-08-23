import pdfplumber
import fitz  # PyMuPDF
from docx import Document
from typing import IO

from .logging_config import get_logger

logger = get_logger(__name__)


def extract_text_from_pdf(pdf_file: IO) -> str:
    """Try PyMuPDF first, fallback to pdfplumber.

    Args:
        pdf_file: file-like object opened in binary mode.

    Returns:
        Extracted text or empty string on failure.
    """
    text = ""

    # First: PyMuPDF
    try:
        file_bytes = pdf_file.read()
        pdf_file.seek(0)  # reset pointer so other libs can read
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                page_text = page.get_text("text")
                if page_text:
                    text += page_text + "\n"
        if len(text.strip()) > 400:  # if PyMuPDF extracted enough
            logger.debug("Extracted PDF via PyMuPDF; length=%d", len(text))
            return text.strip()
    except Exception as exc:  # pragma: no cover - depends on external files
        logger.exception("PyMuPDF extraction failed: %s", exc)

    # Fallback: pdfplumber
    try:
        pdf_file.seek(0)
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        logger.debug("Extracted PDF via pdfplumber; length=%d", len(text))
        return text.strip()
    except Exception as exc:  # pragma: no cover
        logger.exception("pdfplumber extraction failed: %s", exc)
        return ""


def extract_text_from_docx(docx_file: IO) -> str:
    try:
        doc = Document(docx_file)
        text = "\n".join([p.text for p in doc.paragraphs])
        logger.debug("Extracted DOCX; length=%d", len(text))
        return text
    except Exception as exc:
        logger.exception("DOCX extraction failed: %s", exc)
        return ""


def extract_text_from_txt(txt_file: IO) -> str:
    try:
        content = txt_file.read()
        if isinstance(content, bytes):
            text = content.decode("utf-8", errors="replace")
        else:
            text = str(content)
        logger.debug("Extracted TXT; length=%d", len(text))
        return text
    except Exception as exc:
        logger.exception("TXT extraction failed: %s", exc)
        return ""
