import re
from pypdf import PdfReader

def load_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)

    pages = []

    for page in reader.pages:
        page_text = page.extract_text() or ""

        # Remove excessive whitespace
        page_text = re.sub(r"\s+", " ", page_text)

        pages.append(page_text)

    return "\n".join(pages)