from pydantic import BaseModel,Field
from typing import List

class Citation(BaseModel):
    doc_name: str
    chunk_id: int
    text_snippet: str

class AnswerOutput(BaseModel):
    question: str = Field(..., description="Question asked")
    answer: str =Field(..., description="Answer to the question with citation of doc_name eg. Paris is in France [cities_of_france.pdf,chunk 0]")
