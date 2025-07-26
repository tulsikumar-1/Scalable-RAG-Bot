from pydantic import BaseModel
from typing import List

class Citation(BaseModel):
    doc_name: str
    chunk_id: int
    text_snippet: str

class AnswerOutput(BaseModel):
    answer: str
    citations: List[Citation]
