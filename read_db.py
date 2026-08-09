from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def inspect_vector_db(db_path: str = "./chroma_db"):
    print("Loading vector database from disk...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )

    # Fetch stored documents and metadata from the underlying collection
    collection = vectorstore._collection
    data = collection.get(include=['documents', 'metadatas'])
    
    documents = data.get('documents', [])
    metadatas = data.get('metadatas', [])

    print(f"\n--- Total Stored Chunks: {len(documents)} ---")
    
    # Display the first 3 chunks as a preview
    for i, (doc, meta) in enumerate(zip(documents[:3], metadatas[:3]), 1):
        print(f"\n[Chunk #{i}]")
        print(f"Source: {meta.get('source', 'N/A')}")
        print(f"Content Preview: {doc[:200]}...")
        print("-" * 50)

if __name__ == "__main__":
    inspect_vector_db()