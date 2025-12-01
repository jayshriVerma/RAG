import os
from dotenv import load_dotenv
from openai import OpenAI
import faiss
import numpy as np

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Step 1: Knowledge base documents
documents = [
   "which process should i use for legal document knowledge base CAG or RAG?",
    "The knowledge base in creating the law research assistant is dynamic with the new content been added all the time",
    "Python is a versatile programming language.",
    "Pandas is a powerful data manipulation library.",
]

# Step 2: Create embeddings for documents
embeddings = []
for doc in documents:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=doc
    )
    embeddings.append(response.data[0].embedding)

embeddings = np.array(embeddings).astype("float32")

# Step 3: Build FAISS index
d = embeddings.shape[1]  # embedding dimension
index = faiss.IndexFlatL2(d)
index.add(embeddings)

# Step 4: User query
query = "What tool can I use for Augmented generation for legal document?"
query_embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input=query
).data[0].embedding
query_vector = np.array([query_embedding]).astype("float32")

# Step 5: Retrieve top-k documents
k = 2
distances, indices = index.search(query_vector, k)
retrieved_docs = [documents[i] for i in indices[0]]

print("Query:", query)
print("\nRetrieved context:")
for doc in retrieved_docs:
    print("-", doc)

# Step 6: Ask GPT-4o with retrieved context
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
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}],
)

print("\nGPT-4o Answer:")
print(response.choices[0].message.content)

# output:
# Query: What tool can I use for Augmented generation for legal document?

# Retrieved context:
# - which process should i use for legal document knowledge base CAG or RAG?
# - The knowledge base in creating the law research assistant is dynamic with the new content been added all the time

# GPT-4o Answer:
# For augmented generation of legal documents, you can use Retrieval-Augmented Generation (RAG).
# RAG is particularly well-suited for dynamic knowledge bases where new content is continuously added.
# This method effectively combines retrieval mechanisms with generation capabilities, allowing it to pull relevant information from a constantly updating dataset and produce coherent, contextually accurate responses.
# RAG can leverage the extensive and evolving data typical of legal documents to aid in generating responses or analyses based on the most current information available.