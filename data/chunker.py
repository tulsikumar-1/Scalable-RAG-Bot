#chunker.py
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(text: str, max_length=500, overlap=50) -> list[str]:
    """
    Split text into chunks using LangChain's RecursiveCharacterTextSplitter,
    which tries to avoid cutting sentences.

    :param text: full text string
    :param max_length: max chunk size in characters
    :param overlap: number of characters to overlap between chunks
    :return: list of text chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_length,
        chunk_overlap=overlap,
        separators=["\n\n", ".", "!", "?"]
    )
    chunks = splitter.split_text(text)
    return chunks

