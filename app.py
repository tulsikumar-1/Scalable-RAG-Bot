import os
import streamlit as st
from data.loader import load_pdf
from data.chunker import chunk_text
from embeddings.embedder import Embedder
from storage.database import Database
from index.faiss_index import FaissIndex
from retrieval.retriever import RetrievalQA
from llm.llm_model import OnlineLLM
from utils import show_error

DOCUMENTS_DIR = "./database"
EMBEDDING_DIM = 384
FAISS_INDEX_PATH = "faiss_index.index"

@st.cache_resource
def get_embedder():
    return Embedder()

@st.cache_resource
def get_database():
    return Database()

@st.cache_resource
def get_faiss_index():
    index = FaissIndex(dim=EMBEDDING_DIM)
    # Optional: Load index if exists
    if os.path.exists(FAISS_INDEX_PATH):
        index.load(FAISS_INDEX_PATH)
    return index

@st.cache_resource
def get_llm():
    #return OnlineLLM('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')
    return OnlineLLM('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')

@st.cache_resource
def get_qa_system():
    embedder = get_embedder()
    db = get_database()
    index = get_faiss_index()
    qa = RetrievalQA(embedder, index, db)

    # Check DB & build index from all stored chunks
    all_chunks = db.get_all_chunks()
    if all_chunks:
        qa.build_index(all_chunks)
    return qa

def process_document(file_name, file_bytes, db, embedder):
    try:
        doc = db.get_document_by_name(file_name)
        if doc is None:
            text = load_pdf(file_bytes)
            chunks = chunk_text(text)
            doc = db.add_document(file_name)
            embeddings = embedder.embed(chunks)
            db.add_chunks(doc.id, chunks, embeddings)
            st.success(f"Processed and stored '{file_name}' with {len(chunks)} chunks.")
        else:
            st.info(f"Document '{file_name}' already exists in the database.")
    except Exception as e:
        show_error(f"Failed to process {file_name}: {str(e)}")

def process_existing_documents():
    db = get_database()
    embedder = get_embedder()
    files = [f for f in os.listdir(DOCUMENTS_DIR) if f.endswith(".pdf")]

    for file_name in files:
        file_path = os.path.join(DOCUMENTS_DIR, file_name)
        with open(file_path, "rb") as f:
            process_document(file_name, f, db, embedder)

def main():
    st.title("📚 Customizable RAG Bot")

    # Process any new PDFs in the database folder
    process_existing_documents()

    # Upload single PDF
    uploaded_pdf = st.file_uploader("Upload a PDF document", type=["pdf"])
    if uploaded_pdf:
        db = get_database()
        embedder = get_embedder()
        process_document(uploaded_pdf.name, uploaded_pdf, db, embedder)

        # Rebuild index after upload
        all_chunks = db.get_all_chunks()
        index = get_faiss_index()
        retriever = RetrievalQA(embedder, index, db)
        retriever.build_index(all_chunks)
        index.save(FAISS_INDEX_PATH)
        st.success("Index updated with uploaded document.")

    # Load QA system
    qa = get_qa_system()
    llm = get_llm()

    # Question interface
    question = st.text_input("Ask a question about the documents:")
    if question:
        with st.spinner("Generating answer..."):
            try:
                answer_obj = qa.query(question, llm)
                st.markdown(f"**Answer:** {answer_obj.answer}")
                if hasattr(answer_obj, "citations"):
                    st.markdown("**Citations:**")
                    for c in answer_obj.citations:
                        st.markdown(f"- **{c['doc_name']}** (Chunk ID: {c['chunk_id']}):\n> {c['text']}")
            except Exception as e:
                show_error(str(e))

if __name__ == "__main__":
    if not os.path.exists(DOCUMENTS_DIR):
        os.makedirs(DOCUMENTS_DIR)
    main()
