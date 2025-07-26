from pydantic import ValidationError
from schemas.output_schema import AnswerOutput
from llm.prompt_template import get_prompt_template

# This is a stub to simulate an offline LLM.
# Replace with real offline LLM inference (e.g. Mistral, GPT4All)

class OfflineLLM:
    def __init__(self):
        pass

    def generate_answer(self, question: str, retrieved_chunks: list[dict]) -> AnswerOutput:
        """
        Generate answer from LLM given question and retrieved chunks.
        """
        # Prepare prompt text
        chunk_texts = [chunk["text"] for chunk in retrieved_chunks]
        prompt = get_prompt_template(question, chunk_texts)

        # Here you would call your offline LLM, but we simulate an answer:
        simulated_answer = "This is a simulated answer based on the retrieved documents."
        simulated_citations = [
            {
                "doc_name": chunk["doc_name"],
                "chunk_id": chunk["chunk_id"],
                "text_snippet": chunk["text"][:100] + "..."
            }
            for chunk in retrieved_chunks
        ]

        # Validate output format using pydantic
        try:
            answer_output = AnswerOutput(
                answer=simulated_answer,
                citations=simulated_citations
            )
            return answer_output
        except ValidationError as e:
            raise RuntimeError(f"LLM output validation failed: {e}")
