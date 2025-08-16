import pdfplumber
import fitz  # PyMuPDF
from docx import Document

def extract_text_from_pdf(pdf_file):
    """Try PyMuPDF first, fallback to pdfplumber"""
    text = ""

    # First: PyMuPDF
    try:
        file_bytes = pdf_file.read()
        pdf_file.seek(0)  # reset pointer so other libs can read
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
        if len(text.strip()) > 400:  # if PyMuPDF extracted enough
            return text.strip()
    except Exception:
        pass

    # Fallback: pdfplumber
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    except Exception:
        return ""
    

def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    return "\n".join([p.text for p in doc.paragraphs])


def extract_text_from_txt(txt_file):
    return txt_file.read().decode("utf-8")
