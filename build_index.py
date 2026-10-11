from app.excel_loader import load_hr_dataset
from app.embeddings import EmbeddingModel
from app.vector_store import VectorStore


DATASET_PATH = "data/documents/HRDataset_v13.csv.xls"

INDEX_PATH = "data/index/faiss.index"

DOCUMENTS_PATH = "data/index/documents.pkl"


print("Loading dataset...")

documents = load_hr_dataset(
    DATASET_PATH
)

print(
    f"Loaded {len(documents)} documents"
)


print("\nLoading embedding model...")

embedding_model = EmbeddingModel()


print("\nCreating embeddings...")

embeddings = embedding_model.embed_documents(
    documents
)


print("\nBuilding FAISS index...")

vector_store = VectorStore(
    embeddings,
    documents
)


print("\nSaving index...")

vector_store.save(
    INDEX_PATH,
    DOCUMENTS_PATH
)


print("\nDone!")