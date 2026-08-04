import os
import requests
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

def fetch_github_issues(repo_owner: str, repo_name: str, max_issues: int = 30) -> list[Document]:
    """Fetches closed/open issues from a GitHub repository, excluding PRs."""
    github_token = os.getenv("GITHUB_TOKEN")
    headers = {
        "Accept": "application/vnd.github+json",
        **({"Authorization": f"Bearer {github_token}"} if github_token else {})
    }
    
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues"
    params = {"state": "all", "per_page": max_issues}
    
    response = requests.get(url, headers=headers, params=params, timeout=10)
    response.raise_for_status()

    documents = []
    for issue in response.json():
        # Exclude Pull Requests (GitHub issues API includes PRs by default)
        if "pull_request" in issue:
            continue
            
        title = issue.get("title", "")
        body = issue.get("body") or ""
        
        documents.append(
            Document(
                page_content=f"Issue Title: {title}\n\nContent:\n{body}",
                metadata={
                    "source": issue.get("html_url", ""),
                    "issue_number": issue.get("number")
                }
            )
        )
        
    return documents

def main():
    print("🚀 [1/3] Fetching GitHub Issues...")
    docs = fetch_github_issues("huggingface", "peft", max_issues=30)
    print(f"✅ Fetched {len(docs)} pure issues (PRs filtered out).")

    print("✂️ [2/3] Chunking Documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    print(f"✅ Generated {len(chunks)} text chunks.")

    print("🧠 [3/3] Generating Embeddings and Indexing into ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("🎉 Vector database built successfully at './chroma_db'.")

if __name__ == "__main__":
    main()