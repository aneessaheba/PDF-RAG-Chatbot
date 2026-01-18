from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
import logging

# Enable logging to see generated queries and reranking
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def setup_rag_with_reranking():
    """Setup RAG system with MultiQuery retrieval + FlashRank reranking using LangChain."""

    print("Setting up RAG system with MultiQuery + FlashRank Reranking...")

    # Initialize embeddings (same model used for indexing)
    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")

    # Load the existing ChromaDB
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    print(f"Loaded vector store with {vectorstore._collection.count()} documents")

    # Initialize Ollama LLM
    llm = Ollama(
        model="llama3.2:1b",
        temperature=0.7
    )

    # Create base retriever with higher k for reranking
    base_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 10}  # Retrieve more documents initially for reranking
    )

    # Create MultiQueryRetriever
    print("Setting up MultiQuery retriever...")
    multiquery_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )

    # Initialize FlashRank reranker as a document compressor
    print("Loading FlashRank reranker...")
    compressor = FlashrankRerank(
        model="ms-marco-MiniLM-L-12-v2",  # Lightweight reranking model
        top_n=4  # Return top 4 documents after reranking
    )
    print("FlashRank reranker loaded")

    # Wrap the multiquery retriever with contextual compression (reranking)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=multiquery_retriever
    )

    # Create a custom prompt template for answering
    template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer: """

    prompt = ChatPromptTemplate.from_template(template)

    # Create RAG chain using LCEL with multiquery + reranking
    rag_chain = (
        {"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("RAG system with MultiQuery + FlashRank Reranking ready!\n")
    print("Pipeline: MultiQuery (generates variations) -> Retrieve (from vector DB) -> FlashRank Rerank (best 4) -> Answer\n")

    # Return the chain and the compression retriever
    return rag_chain, compression_retriever

if __name__ == "__main__":
    rag_chain, retriever = setup_rag_with_reranking()
    print("RAG chain with FlashRank reranking created successfully!")
    print("Import this module in other scripts to use the RAG system.")
