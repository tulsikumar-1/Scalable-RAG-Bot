import re
from typing import List

def split_into_sentences(text: str) -> List[str]:
    """
    Naive sentence splitter using regular expressions.
    Handles '.', '!', '?' as sentence terminators.
    """
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    return sentence_endings.split(text.strip())

def chunk_text(text: str, max_length=500, overlap=1) -> List[str]:
    """
    Split text into chunks based on sentence boundaries.
    Allows overlapping sentences between chunks.
    
    :param text: The full text string.
    :param max_length: Target maximum length of each chunk (soft limit).
    :param overlap: Number of sentences to overlap between chunks.
    :return: List of text chunks.
    """
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = []
    current_length = 0

    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        if current_length + len(sentence) <= max_length or not current_chunk:
            current_chunk.append(sentence)
            current_length += len(sentence) + 1  # +1 for space
            i += 1
        else:
            chunks.append(" ".join(current_chunk))
            # Move back by `overlap` sentences
            i = max(0, i - overlap)
            current_chunk = []
            current_length = 0

    # Add last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
