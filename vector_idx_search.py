import os
from dotenv import load_dotenv
from openai import OpenAI
import faiss
import numpy as np

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Step 1: Some example texts
documents = [
    "Jupyter notebooks are great for data science.",
    "Python is a versatile programming language.",
    "Pandas is a powerful data manipulation library.",
    "I love playing chess on weekends.",
    "Machine learning enables predictive models.",
]
# Step 2: Get embeddings from OpenAI
embeddings = []
for doc in documents:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=doc
    )
    embeddings.append(response.data[0].embedding)

embeddings = np.array(embeddings).astype("float32")

# Step 3: Build a FAISS index
d = embeddings.shape[1]  # embedding dimension
index = faiss.IndexFlatL2(d)  # L2 = Euclidean distance
index.add(embeddings)  # add all document vectors
print(f"Number of vectors in the index: {index.ntotal}")

# Step 4: Query example
query = "Which tool helps with interactive coding for data analysis?"
query_embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input=query
).data[0].embedding

query_vector = np.array([query_embedding]).astype("float32")

# Step 5: Search top-2 matches
k = 3
distances, indices = index.search(query_vector, k)

print("Query:", query)
print("\nTop matches:")
for idx, dist in zip(indices[0], distances[0]):
    print(f"- {documents[idx]} (score={dist:.4f})")

# Output:
# Number of vectors in the index: 6
# Query: Which tool helps with interactive coding for data analysis?

# Top matches:
# - Jupyter notebooks are great for data science. (score=1.0886)
# - Pandas is a powerful data manipulation library. (score=1.2123)
# - Python is a versatile programming language. (score=1.4193)

# Each document → converted into a vector (embedding).

# FAISS stored these vectors in an index.

# Converting query into vector embedding.

# FAISS quickly found the closest embeddings (semantic similarity).



