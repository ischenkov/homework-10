from pathlib import Path
import pickle

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Settings

settings = Settings()
DATA_DIR = Path(__file__).parent / "data"
INDEX_DIR = Path(__file__).parent / "index"


def load_documents() -> list:
    documents = []
    data_path = DATA_DIR

    if not data_path.exists():
        data_path.mkdir(parents=True)
        print("Created empty data/ directory. Add PDF or TXT files and run again.")
        return []

    for file_path in data_path.glob("*"):
        if file_path.suffix.lower() == ".pdf":
            try:
                loader = PyPDFLoader(str(file_path))
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                print(f"Warning: Could not load PDF {file_path}: {e}")
        elif file_path.suffix.lower() in (".txt", ".md"):
            try:
                loader = TextLoader(str(file_path), encoding="utf-8")
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")

    return documents


def chunk_documents(documents: list):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def main():
    print("Loading documents from", DATA_DIR)
    documents = load_documents()

    if not documents:
        print("No documents found. Add PDF or TXT files to ./data/ and try again.")
        return

    print(f"Loaded {len(documents)} document(s)")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=settings.api_key.get_secret_value(),
    )

    print("Creating embeddings and FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))
    with open(INDEX_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print(f"Index saved to {INDEX_DIR}")


if __name__ == "__main__":
    main()
