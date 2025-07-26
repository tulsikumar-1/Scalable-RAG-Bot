import streamlit as st
from data.loader import load_pdf
from data.chunker import chunk_text
from embeddings.embedder import Embedder
from storage.database import Database
from index.faiss_index import FaissIndex
from retrieval.retriever import RetrievalQA
from llm.llm_model import OfflineLLM
from utils import show_error

@st.cache_resource
def get_embedder():
    return Embedder()

@st.cache_resource
def get_database():
    return Database()

@st.cache_resource
def get_faiss_index():
    # Embedding dimension for all-MiniLM-L6-v2 is 384
    return FaissIndex(dim=384)

@st.cache_resource
def get_qa_system():
    embedder = get_embedder()
    db = get_database()
    index = get_faiss_index()
    qa = RetrievalQA(embedder, index, db)
    qa.build_index()
    return qa

@st.cache_resource
def get_llm():
    return OfflineLLM()

def main():
    st.title("Offline RAG System")

    uploaded_pdf = st.file_uploader("Upload a PDF document", type=["pdf"])
    if uploaded_pdf:
        text = load_pdf(uploaded_pdf)
        chunks = chunk_text(text)
        st.success(f"Extracted {len(chunks)} chunks from PDF.")

        db = get_database()
        embedder = get_embedder()

        try:
            doc = db.add_document(uploaded_pdf.name)
            embeddings = embedder.embed(chunks)
            db.add_chunks(doc.id, chunks, embeddings)
            st.success("Saved document and chunks.")
        except Exception as e:
            show_error(str(e))
            return

        index = get_faiss_index()
        qa = RetrievalQA(embedder, index, db)
        qa.build_index()
        st.success("FAISS index updated.")

    qa = get_qa_system()
    llm = get_llm()

    question = st.text_input("Ask a question about the uploaded documents:")
    if question:
        with st.spinner("Generating answer..."):
            try:
                answer_obj = qa.query(question, llm)
                st.markdown(f"**Answer:** {answer_obj.answer}")
                st.markdown("**Citations:**")
                for c in answer_obj.citations:
                    st.markdown(f"- Doc: {c.doc_name}, Chunk ID: {c.chunk_id}\n> {c.text_snippet}")
            except Exception as e:
                show_error(str(e))

if __name__ == "__main__":
    main()
