import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# LangChain v0.2+ / v0.3+ direct imports (clears Pyright missing import warnings)
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

class RAGEngine:
    def __init__(self, db_path="./chroma_db"):
        # 1. Ensure HF Token is set globally in environment variables
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

        # 2. Initialize Embeddings & Vector Store
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        # Track latest query embedding and retrieved docs for visualization
        self.latest_query = None
        self.latest_query_embedding = None
        self.latest_retrieved_docs = []

        # 3. Define LLM Endpoint
        llm_endpoint = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
            huggingfacehub_api_token=hf_token,
            temperature=0.2,
            max_new_tokens=512
        )
        self.chat_model = ChatHuggingFace(llm=llm_endpoint)

        # 4. Build History-Aware Retriever Chain
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        self.history_aware_retriever = create_history_aware_retriever(
            self.chat_model, self.retriever, contextualize_q_prompt
        )

        # 5. Build Document Combination QA Chain
        system_prompt = (
            "You are an expert AI software engineer analyzing GitHub repository issues. "
            "You must ALWAYS communicate and respond STRICTLY in ENGLISH, regardless of the language used by the user. "
            "Use the provided context to answer the user's question clearly, concisely, and accurately in English. "
            "Preserve technical terms and code identifiers.\n\n"
            "Context:\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(self.chat_model, qa_prompt)

        # 6. Final Retrieval RAG Chain
        self.rag_chain = create_retrieval_chain(self.history_aware_retriever, question_answer_chain)

    def ask(self, query: str, chat_history: list = None):
        """Executes the LCEL RAG chain with conversational memory and retrieves context."""
        # Embed and store the query vector for visualization
        self.latest_query = query
        self.latest_query_embedding = self.embeddings.embed_query(query)

        formatted_history = []
        if chat_history:
            for msg in chat_history[-4:]:
                if msg["role"] == "user":
                    formatted_history.append(("human", msg["content"]))
                elif msg["role"] == "assistant":
                    formatted_history.append(("ai", msg["content"]))

        # Execute Chain
        response = self.rag_chain.invoke({
            "input": query,
            "chat_history": formatted_history
        })

        # Store retrieved context documents for 2D visualization match
        self.latest_retrieved_docs = response["context"]

        return {
            "answer": response["answer"],
            "sources": [doc.metadata.get("source", "N/A") for doc in response["context"]],
            "raw_docs": response["context"]
        }