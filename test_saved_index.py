from app.vector_store import VectorStore


INDEX_PATH = "data/index/faiss.index"

DOCUMENTS_PATH = "data/index/documents.pkl"


vector_store = VectorStore.load(
    INDEX_PATH,
    DOCUMENTS_PATH
)


print(
    f"Loaded {len(vector_store.documents)} documents"
)

print(
    "FAISS vectors:",
    vector_store.index.ntotal
)