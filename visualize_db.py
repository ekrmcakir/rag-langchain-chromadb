import numpy as np
import plotly.express as px
import umap
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def visualize_vector_space(db_path: str = "./chroma_db"):
    print("1. Loading embeddings and ChromaDB vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )

    print("2. Fetching embeddings and raw document chunks...")
    # Retrieve raw vectors and corresponding text contents
    data = vectorstore._collection.get(include=['embeddings', 'documents'])
    raw_embeddings = np.array(data['embeddings'])
    
    # Truncate text documents for clean hover text on Plotly
    documents = [doc[:100].replace("\n", " ") + "..." for doc in data['documents']]

    if len(raw_embeddings) == 0:
        print("Error: No vectors found in ChromaDB. Run main.py first.")
        return

    print(f"3. Reducing dimensions for {len(raw_embeddings)} vectors using UMAP (384D -> 2D)...")
    reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, random_state=42)
    embedding_2d = reducer.fit_transform(raw_embeddings)

    print("4. Rendering 2D Scatter Plot...")
    fig = px.scatter(
        x=embedding_2d[:, 0],
        y=embedding_2d[:, 1],
        hover_name=documents,
        title="ChromaDB Vector Space Visualization (UMAP)",
        labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2'}
    )
    
    fig.update_traces(marker=dict(size=10, opacity=0.8))
    fig.show()

if __name__ == "__main__":
    visualize_vector_space()