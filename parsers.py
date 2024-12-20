"""
Parsers are needed to chunk documents into usable pieces.

Language models are improving, but their text limit is still finite.

"""

from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO

def parse_pdf(pdf_file):
    """
    Extracts text from a PDF file, page by page.
    
    Parameters:
    - pdf_file: A file-like object (e.g., BytesIO, UploadedFile) containing the PDF data.
    
    Returns:
    - A list where each element contains the text of a respective PDF page.
    """

    texts = []

     # Extract text from each page
    
    for page_number, page_layout in enumerate(PDFPage.get_pages(pdf_file, caching=True, check_extractable=True, maxpages=100)):
        pdf_file.seek(0)
        text = extract_text(pdf_file, page_numbers=[page_number], laparams=LAParams())
        if text.strip():
            texts.append(text)

    return texts