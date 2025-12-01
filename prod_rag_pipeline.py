import os
import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from pathlib import Path
from typing import List

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class AdvancedRAG:
    """A Retrieval-Augmented Generation (RAG) system that:
        - Loads documents from txt and pdf files
        - Chunks large documents
        - Embeds chunks using OpenAI
        - Indexes embeddings with FAISS
        - Retrieves relevant chunks for a query
        - Generates answers using GPT-4o based on retrieved context
    """
    def __init__(
        self,
        embedding_model="text-embedding-3-small",
        llm_model="gpt-4o",
        chunk_size=100, # Each chunk ≈ 100 characters
        overlap=20, # 20 characters overlap between chunks, perfect for small files
    ):
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.chunk_size = chunk_size
        self.overlap = overlap

        self.documents = []
        self.chunks = []
        self.embeddings = None
        self.index = None

    # ------------------------
    # FILE LOADING
    # ------------------------

    def load_txt(self, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            self.documents.append(f.read())

    def load_pdf(self, file_path: str):
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        self.documents.append(text)

    def load_folder(self, folder_path: str):
        for file in Path(folder_path).iterdir():
            if file.suffix == ".txt":
                self.load_txt(file)
            elif file.suffix == ".pdf":
                self.load_pdf(file)

    # ------------------------
    # CHUNKING
    # ------------------------

    def _chunk_text(self, text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += self.chunk_size - self.overlap
        return chunks

    def prepare_chunks(self):
        self.chunks = []
        for doc in self.documents:
            self.chunks.extend(self._chunk_text(doc))

    # ------------------------
    # EMBEDDING + INDEXING
    # ------------------------

    def _embed(self, text: str):
        response = client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        return np.array(response.data[0].embedding, dtype="float32")

    def build_index(self):
        if not self.documents:
            raise ValueError("No documents loaded.")

        self.prepare_chunks()
        self.embeddings = np.array([self._embed(c) for c in self.chunks])

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

    # ------------------------
    # RETRIEVAL
    # ------------------------

    def retrieve(self, query: str, k=4):
        query_emb = self._embed(query).reshape(1, -1)
        distances, indices = self.index.search(query_emb, k)
        return [self.chunks[i] for i in indices[0]]

    # ------------------------
    # RAG QUESTION ANSWERING
    # ------------------------

    def ask(self, query: str, k=4):
        retrieved_chunks = self.retrieve(query, k)
        context = "\n\n".join(retrieved_chunks)

        prompt = f"""
        You are an AI assistant. Answer strictly based on the context.

        Context:
        {context}

        Question:
        {query}

        Answer:
        """

        response = client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.choices[0].message.content
if __name__ == "__main__":
    # Example usage
    rag = AdvancedRAG()

    # Load documents from a folder
    # rag.load_folder("knowledge_base/")
    rag.load_txt("notes.txt")
    # rag.load_pdf("machine_learning.pdf")

    # Build the FAISS index
    rag.build_index()

    # Ask a question
    query = "What tool can I use for Augmented generation for legal document?"
    answer = rag.ask(query)
    print(len(rag.documents))     # should be > 0
    print(len(rag.chunks))        # should be > 1
    print(rag.embeddings.shape)  # should NOT be empty
    print(answer)
# output:
# 1
# 9
# (9, 1536)
# You should use RAG (Retrieval Augmented Generation) for augmented generation in legal documents.
# This is the same architecture used by:
# LangChain, LlamaIndex, Chat-with-PDF apps, Enterprise knowledge assistants
# Only differences in production:
# Use Pinecone / Weaviate instead of FAISS
# Add streaming responses
# Add authentication + caching
# =================================
# for medium files(5-20 pages)
# chunk_size=300
# overlap=50
# for large files(100+ pages)
# chunk_size=800
# overlap=150
#overlap=20 means 20 characters from the previous chunk are repeated in the next chunk to preserve context.
    
    
    