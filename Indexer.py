import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

def index_pdfs(filename):
    """Index all PDFs in the current directory using Ollama embeddings and ChromaDB."""

    # Get all PDF files in the current directory
    pdf_files = glob.glob(filename)

    if not pdf_files:
        print("No PDF files found in the current directory.")
        return

    print(f"Found {len(pdf_files)} PDF file(s): {pdf_files}")

    # Initialize embeddings with Ollama using qwen3-embedding:0.6b model
    embeddings = OllamaEmbeddings(
        model="qwen3-embedding:0.6b"
    )

    # Load and process all PDFs
    all_documents = []

    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file}")

        # Load PDF
        loader = PyPDFLoader(pdf_file)
        documents = loader.load()

        print(f"  Loaded {len(documents)} pages")

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        splits = text_splitter.split_documents(documents)
        print(f"  Created {len(splits)} chunks")

        all_documents.extend(splits)

    print(f"\nTotal chunks to index: {len(all_documents)}")

    # Create ChromaDB vector store
    print("\nCreating ChromaDB vector store...")
    vectorstore = Chroma.from_documents(
        documents=all_documents,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    print(f"Successfully indexed {len(all_documents)} chunks into ChromaDB!")
    print(f"Vector store saved to: ./chroma_db")

    return vectorstore

if __name__ == "__main__":
    index_pdfs("Foundations_of_Machine_Learning.pdf")
