import io
from typing import Generator
import fitz  # PyMuPDF


def pdf_to_images(pdf_bytes: bytes, dpi: int = 200) -> Generator[tuple[int, bytes], None, None]:
    """Yield (page_number, png_bytes) for each page in the PDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page_num, page in enumerate(doc, start=1):
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        yield page_num, pix.tobytes("png")
    doc.close()


def pdf_page_count(pdf_bytes: bytes) -> int:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    count = doc.page_count
    doc.close()
    return count
