def get_prompt_template(question: str, retrieved_chunks: list[str],format : str) -> str:
    """
    Create prompt for LLM based on question and retrieved chunks.
    """
    context = retrieved_chunks
    prompt = """
        Use the following context to answer the question.\n\n
        Context:\n{context}\n\n
        Question:\n{question}\n\n
        output format instruction:{format}\n
        Answer with citations to the source documents with doc_name and chunk
      
"""
    return prompt.format(context=context, question=question,format=format)