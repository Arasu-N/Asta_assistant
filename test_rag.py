from app.excel_loader import load_hr_dataset
from app.embeddings import EmbeddingModel
from app.vector_store import VectorStore
from app.groq_generator import GroqGenerator


# ============================================
# 1. LOAD DATA
# ============================================

dataset_path = "data/documents/HRDataset_v13.csv.xls"

print("Loading HR dataset...")

documents = load_hr_dataset(dataset_path)

print(f"Loaded {len(documents)} documents")


# ============================================
# 2. CREATE EMBEDDINGS
# ============================================

print("\nLoading embedding model...")

embedding_model = EmbeddingModel()

print("\nCreating embeddings...")

embeddings = embedding_model.embed_documents(documents)


# ============================================
# 3. CREATE FAISS VECTOR STORE
# ============================================

print("\nCreating FAISS index...")

vector_store = VectorStore(
    embeddings,
    documents
)


# ============================================
# 4. LOAD GROQ
# ============================================

print("\nConnecting to Groq...")

groq = GroqGenerator()

print("Groq connected!")


# ============================================
# 5. ASK QUESTION
# ============================================

question = input("\nAsk a question about the HR dataset: ")


# ============================================
# 6. EMBED THE QUESTION
# ============================================

query_embedding = embedding_model.embed_query(question)


# ============================================
# 7. RETRIEVE RELEVANT DOCUMENTS
# ============================================

results = vector_store.search(
    query_embedding,
    top_k=5
)


# ============================================
# 8. BUILD CONTEXT
# ============================================

context_parts = []

for result in results:

    context_parts.append(
        result["document"]["text"]
    )

context = "\n\n---\n\n".join(context_parts)


# ============================================
# 9. SEND CONTEXT TO GROQ
# ============================================

answer = groq.generate(
    question,
    context
)


# ============================================
# 10. DISPLAY ANSWER
# ============================================

print("\n================================")
print("RAG ANSWER")
print("================================")

print(answer)