import pdfplumber
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document

def read_pdf(file_path: str) -> str:
    """Reads the PDF file and returns the extracted text."""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def split_text(docs: str, chunk_size: int = 1000, chunk_overlap: int = 0):
    """Splits the given text into chunks."""
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    document = [Document(page_content=docs)]
    return text_splitter.split_documents(document)
