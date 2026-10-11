from app.excel_loader import load_hr_dataset
from app.embeddings import EmbeddingModel


dataset_path = "data/documents/HRDataset_v13.csv.xls"

print("Loading HR dataset...")

documents = load_hr_dataset(dataset_path)

print(f"Loaded {len(documents)} documents")

print("\nCreating embeddings...")

embedding_model = EmbeddingModel()

embeddings = embedding_model.embed_documents(documents)

print("\nEmbedding complete!")
print("Embedding shape:", embeddings.shape)

print("\nFirst embedding:")
print(embeddings[0])