from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.retrievers import BM25Retriever
import logging

# Enable logging to see generated queries and reranking
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def setup_rag_hybrid():
    """
    Setup RAG system with Hybrid Search (BM25 + Semantic) + MultiQuery + FlashRank Reranking.

    Architecture:
    1. BM25 retriever: Uses original query (keyword-based)
    2. Semantic retriever: Wrapped with MultiQuery for query variations
    3. Ensemble: Combines BM25 + MultiQuery-Semantic results
    4. FlashRank: Reranks combined results to top 4
    """

    print("Setting up RAG system with Hybrid Search + MultiQuery + FlashRank Reranking...")

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
        model="llama3.2:3b",
        temperature=0.7
    )

    # Step 1: Create semantic retriever (dense embeddings)
    print("Setting up semantic retriever...")
    semantic_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}  # Retrieve documents using semantic similarity
    )

    # Step 2: Wrap semantic retriever with MultiQuery (generates query variations)
    print("Wrapping semantic retriever with MultiQuery...")
    multiquery_semantic_retriever = MultiQueryRetriever.from_llm(
        retriever=semantic_retriever,
        llm=llm
    )

    # Step 3: Create BM25 retriever (sparse keyword-based)
    print("Setting up BM25 retriever...")
    # Get all documents from vectorstore for BM25 indexing
    all_docs = vectorstore.get()
    from langchain_core.documents import Document
    documents = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(all_docs['documents'], all_docs['metadatas'])
    ]

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 4  # Retrieve top 8 documents
    print(f"BM25 retriever initialized with {len(documents)} documents")

    # Step 4: Create hybrid retriever (Ensemble of BM25 + MultiQuery-Semantic)
    print("Creating hybrid ensemble retriever...")
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, multiquery_semantic_retriever],
        weights=[0.5, 0.5]  # Equal weight to both retrievers
    )

    # Step 5: Add FlashRank reranking on top of hybrid results
    print("Adding FlashRank reranker...")
    compressor = FlashrankRerank(
        model="ms-marco-MiniLM-L-12-v2",
        top_n=4  # Return top 4 documents after reranking
    )

    # Step 6: Wrap hybrid retriever with contextual compression (reranking)
    final_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=hybrid_retriever
    )

    # Create a custom prompt template for answering
    template = """You are a helpful AI assistant. Use ONLY the context below to answer the question.
If the answer is in the context, provide it clearly and concisely.
If the answer is NOT in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    # Create RAG chain using LCEL
    rag_chain = (
        {"context": final_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\nRAG system with Hybrid Search + MultiQuery + FlashRank ready!\n")
    print("Pipeline:")
    print("  1. Query → BM25 (keyword search on original query)")
    print("  2. Query → MultiQuery → Semantic (variations for embedding search)")
    print("  3. Ensemble: Combine BM25 + Semantic results")
    print("  4. FlashRank: Rerank to top 4 most relevant documents")
    print("  5. Generate answer\n")

    # Return the chain and the final retriever
    return rag_chain, final_retriever

if __name__ == "__main__":
    rag_chain, retriever = setup_rag_hybrid()
    print("RAG chain with Hybrid Search + Reranking created successfully!")
    print("Import this module in other scripts to use the RAG system.")
