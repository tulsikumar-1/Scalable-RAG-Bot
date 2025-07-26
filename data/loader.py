import pdfplumber
import tempfile

def load_pdf(file) -> str:
    """
    Extract text from uploaded PDF file.
    :param file: file-like object (uploaded from Streamlit)
    :return: extracted text as string
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        tmp.flush()
        with pdfplumber.open(tmp.name) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
    text = "\n".join(pages)
    return text
