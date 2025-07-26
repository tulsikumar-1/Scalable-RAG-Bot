from sqlalchemy import create_engine, Column, Integer, String, Text, Float, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
import os

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    chunks = relationship("Chunk", back_populates="document")

class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    text = Column(Text, nullable=False)
    embedding = Column(Text, nullable=False)  # store as stringified list or np array bytes base64
    document = relationship("Document", back_populates="chunks")

class Database:
    def __init__(self, db_path="sqlite:///rag_offline.db"):
        self.engine = create_engine(db_path, echo=False, connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def add_document(self, name: str) -> Document:
        session = self.Session()
        doc = session.query(Document).filter(Document.name == name).first()
        if doc:
            session.close()
            raise ValueError(f"Document with name '{name}' already exists.")
        doc = Document(name=name)
        session.add(doc)
        session.commit()
        session.refresh(doc)
        session.close()
        return doc

    def add_chunks(self, document_id: int, chunks: list[str], embeddings) -> None:
        import json
        session = self.Session()
        for i, (chunk_text, emb) in enumerate(zip(chunks, embeddings)):
            emb_str = json.dumps(emb.tolist())
            chunk = Chunk(document_id=document_id, text=chunk_text, embedding=emb_str)
            session.add(chunk)
        session.commit()
        session.close()

    def get_all_chunks(self):
        """
        Returns all chunks as list of dicts with keys:
        id, document_id, text, embedding, document_name
        """
        import json
        session = self.Session()
        results = []
        chunks = session.query(Chunk).all()
        for chunk in chunks:
            emb = json.loads(chunk.embedding)
            results.append({
                "chunk_id": chunk.id,
                "doc_id": chunk.document_id,
                "text": chunk.text,
                "embedding": emb,
                "doc_name": chunk.document.name
            })
        session.close()
        return results
