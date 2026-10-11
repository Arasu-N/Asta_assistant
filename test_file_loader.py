from app.file_loader import load_dataset


file_path = "data/documents/HRDataset_v13.csv.xls"

documents = load_dataset(file_path)

print(f"Loaded {len(documents)} documents")

for document in documents[:3]:
    print("\n--- DOCUMENT ---")
    print(document["text"])