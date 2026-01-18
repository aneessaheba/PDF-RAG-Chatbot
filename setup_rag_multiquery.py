from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import logging

# Enable logging to see generated queries
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def setup_rag_multiquery():
    """Setup RAG system with MultiQuery retriever for improved retrieval."""

    print("Setting up RAG system with MultiQuery technique...")

    # Initialize embeddings (same model used for indexing)
    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")

    # Load the existing ChromaDB
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    print(f"Loaded vector store with {vectorstore._collection.count()} documents")

    # Initialize Ollama LLM (you can change the model)
    llm = Ollama(
        model="llama3.2:1b",  # Change to your preferred model
        temperature=0.7
    )

    # Create base retriever
    base_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}  # Retrieve top 3 per query variation
    )

    # Create MultiQueryRetriever
    # This will generate multiple query variations and retrieve documents for each
    multiquery_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )

    # Create a custom prompt template for answering
    template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer: """

    prompt = ChatPromptTemplate.from_template(template)

    # Create RAG chain using LCEL with MultiQueryRetriever
    rag_chain = (
        {"context": multiquery_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("RAG system with MultiQuery ready!\n")
    print("MultiQuery will generate multiple variations of your question to improve retrieval.\n")

    # Return the chain and multiquery retriever
    return rag_chain, multiquery_retriever

if __name__ == "__main__":
    rag_chain, retriever = setup_rag_multiquery()
    print("MultiQuery RAG chain created successfully!")
    print("Import this module in other scripts to use the RAG system.")
