import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import umap
from rag_chain import RAGEngine

# Configure Streamlit Page Settings
st.set_page_config(page_title="GitHub Issue RAG Assistant", page_icon="🤖", layout="wide")

@st.cache_resource
def load_rag_engine():
    return RAGEngine()

rag = load_rag_engine()

def create_umap_plot(df: pd.DataFrame, has_query: bool) -> go.Figure:
    """Helper function to build the Plotly vector space visualization."""
    fig = go.Figure()

    # Define trace configurations for clean rendering
    styles = {
        'Repository Document': {'color': '#1f77b4', 'size': 9, 'symbol': 'circle', 'opacity': 0.7},
        'Retrieved Top-K Context': {'color': '#2ca02c', 'size': 14, 'symbol': 'square', 'opacity': 0.9},
        'Active User Query': {'color': '#d62728', 'size': 18, 'symbol': 'star', 'opacity': 1.0}
    }

    for category, group in df.groupby('Type'):
        style = styles.get(category, {})
        fig.add_trace(go.Scatter(
            x=group['UMAP 1'], y=group['UMAP 2'],
            mode='markers',
            name=category,
            text=group['Document'],
            marker=dict(size=style.get('size', 10), color=style.get('color', '#1f77b4'), symbol=style.get('symbol', 'circle'), opacity=style.get('opacity', 0.8))
        ))

    # Add distance lines between active query and retrieved documents
    if has_query and 'Retrieved Top-K Context' in df['Type'].values:
        qx, qy = df[df['Type'] == 'Active User Query'].iloc[0][['UMAP 1', 'UMAP 2']]
        retrieved_df = df[df['Type'] == 'Retrieved Top-K Context']

        for i, (_, row) in enumerate(retrieved_df.iterrows()):
            fig.add_trace(go.Scatter(
                x=[qx, row['UMAP 1']], y=[qy, row['UMAP 2']],
                mode='lines',
                name='Vector Distance / Match',
                line=dict(color='#ff7f0e', width=2, dash='dash'),
                showlegend=(i == 0),
                legendgroup='retrieval_lines',
                hoverinfo='none'
            ))

    fig.update_layout(
        title="ChromaDB Vector Embeddings Projection with RAG Matching",
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
        hovermode='closest'
    )
    return fig

# --- Sidebar ---
with st.sidebar:
    st.title("📌 About the Project")
    st.info("Retrieval-Augmented Generation (RAG) system indexing PEFT GitHub issues.")
    
    st.subheader("💡 Example Questions")
    example_questions = [
        "What is the root cause of the delete_adapter issue in MixedModel?",
        "How can we resolve the delete_adapter function error based on the retrieved documents?",
        "What are the common problems reported regarding save_and_load.py?",
        "Which PyTorch or PEFT versions are referenced in the reported issues?"
    ]
    
    for q in example_questions:
        if st.button(q, use_container_width=True):
            st.session_state.example_prompt = q
            st.rerun()

    st.markdown("---")
    if st.button("💬 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- Main Layout ---
st.title("🤖 GitHub Issues RAG Assistant")
tab1, tab2 = st.tabs(["💬 Chat Assistant", "📊 Vector Space Visualization"])

# --- Tab 1: Chat Assistant ---
with tab1:
    st.session_state.setdefault("messages", [])

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle input from text box or sidebar buttons
    prompt = st.chat_input("Ask a question about PEFT repository issues...")
    if st.session_state.get("example_prompt"):
        prompt = st.session_state.pop("example_prompt")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching vector database and generating response..."):
                result = rag.ask(prompt, chat_history=st.session_state.messages)
                st.markdown(result["answer"])
                
                with st.expander("🔍 Retrieved Context Documents"):
                    for i, doc in enumerate(result["raw_docs"], 1):
                        st.markdown(f"**Source Chunk #{i}:**")
                        st.caption(f"{doc.page_content[:300]}...")

        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

# --- Tab 2: Vector Space Map ---
with tab2:
    st.header("ChromaDB Vector Space (2D Projection)")
    st.caption("384-dimensional embeddings reduced to 2D using UMAP algorithm.")
    
    if st.button("Render Vector Space Map"):
        with st.spinner("Processing embeddings and rendering 2D scatter plot..."):
            data = rag.vectorstore._collection.get(include=['embeddings', 'documents'])
            raw_embeddings = list(data['embeddings'])
            raw_docs = data['documents']
            
            retrieved_contents = [doc.page_content for doc in rag.latest_retrieved_docs] if rag.latest_retrieved_docs else []

            categories = ["Retrieved Top-K Context" if doc in retrieved_contents else "Repository Document" for doc in raw_docs]
            documents = [f"{doc[:100].replace('\n', ' ')}..." for doc in raw_docs]

            has_query = rag.latest_query_embedding is not None
            if has_query:
                raw_embeddings.append(rag.latest_query_embedding)
                documents.append(f"User Query: {rag.latest_query}")
                categories.append("Active User Query")

            # Dimensionality reduction
            reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, random_state=42)
            embedding_2d = reducer.fit_transform(np.array(raw_embeddings))

            df = pd.DataFrame({
                'UMAP 1': embedding_2d[:, 0],
                'UMAP 2': embedding_2d[:, 1],
                'Document': documents,
                'Type': categories
            })

            fig = create_umap_plot(df, has_query)
            st.plotly_chart(fig, use_container_width=True)