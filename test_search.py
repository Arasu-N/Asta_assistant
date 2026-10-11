from app.excel_loader import load_hr_dataset
from app.embeddings import EmbeddingModel
from app.vector_store import VectorStore


# -----------------------------
# 1. Load dataset
# -----------------------------

dataset_path = "data/documents/HRDataset_v13.csv.xls"

print("Loading dataset...")

documents = load_hr_dataset(dataset_path)

print(f"Loaded {len(documents)} documents")


# -----------------------------
# 2. Create embeddings
# -----------------------------

print("\nLoading embedding model...")

embedding_model = EmbeddingModel()

print("\nCreating document embeddings...")

embeddings = embedding_model.embed_documents(documents)


# -----------------------------
# 3. Create FAISS index
# -----------------------------

print("\nCreating FAISS vector store...")

vector_store = VectorStore(
    embeddings,
    documents
)


# -----------------------------
# 4. Ask a question
# -----------------------------

query = "Who works in Production?"

print(f"\nSearching for: {query}")

query_embedding = embedding_model.embed_query(query)


# -----------------------------
# 5. Search
# -----------------------------

results = vector_store.search(
    query_embedding,
    top_k=5
)


# -----------------------------
# 6. Display results
# -----------------------------

print("\n===== SEARCH RESULTS =====")

for i, result in enumerate(results, start=1):

    print(f"\n--- Result {i} ---")
    print(f"Distance: {result['distance']:.4f}")

    print(result["document"]["text"])