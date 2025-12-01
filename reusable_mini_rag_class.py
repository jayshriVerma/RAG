import os
import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # loads .env file with OPENAI_API_KEY
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class MiniRAG:
    """Call rag.ask("your question") and it:
        Embeds the query
        Finds the most relevant docs via FAISS
        Sends both to GPT-4o
        Returns a grounded answer
    """
    def __init__(self, documents, embedding_model="text-embedding-3-small", llm_model="gpt-4o"):
        self.documents = documents
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.index = None
        self.embeddings = None
        self._build_index()

    def _embed_text(self, text):
        """Get embedding from OpenAI API"""
        response = client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return np.array(response.data[0].embedding, dtype="float32")

    def _build_index(self):
        """Embed all docs and build FAISS index"""
        self.embeddings = np.array([self._embed_text(doc) for doc in self.documents])
        d = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(self.embeddings)

    def retrieve(self, query, k=2):
        """Retrieve top-k docs relevant to query"""
        query_emb = self._embed_text(query).reshape(1, -1)
        distances, indices = self.index.search(query_emb, k)
        return [self.documents[i] for i in indices[0]]

    def ask(self, query, k=2):
        """Answer query using retrieved docs + GPT-4o"""
        retrieved_docs = self.retrieve(query, k)
        context = "\n".join(retrieved_docs)
        prompt = f"""
        You are an AI assistant. Use the context below to answer the question.

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
    documents = [
        "which process should i use for legal document knowledge base CAG or RAG?",
        "Jupyter notebooks are great for data science.",
        "Python is a versatile programming language.",
        "Pandas is a powerful data manipulation library.",
    ]

    rag = MiniRAG(documents)

    query = "What tool can I use for Augmented generation for legal document?"
    answer = rag.ask(query, k=2)

    print("Query:", query)
    print("\nGPT-4o Answer:")
    print(answer)

# output:
# Query: What tool can I use for Augmented generation for legal document?

# GPT-4o Answer:
# For augmented generation of legal documents, you can use Retrieval-Augmented Generation (RAG).