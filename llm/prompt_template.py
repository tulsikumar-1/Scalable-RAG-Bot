def get_prompt_template(question: str, retrieved_chunks: list[str]) -> str:
    """
    Create prompt for LLM based on question and retrieved chunks.
    """
    context = "\n\n".join(retrieved_chunks)
    prompt = (
        f"Use the following context to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        f"Answer with citations to the source documents."
    )
    return prompt
