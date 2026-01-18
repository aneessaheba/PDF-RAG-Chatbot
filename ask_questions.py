#from setup_rag import setup_rag  # Basic RAG
#from setup_rag_multiquery import setup_rag_multiquery  # MultiQuery RAG for improved retrieval
#from setup_rag_rerank import setup_rag_with_reranking  # MultiQuery + FlashRank Reranking
from setup_rag_hybrid import setup_rag_hybrid  # Hybrid Search + MultiQuery + FlashRank

# Global variables to cache the RAG system (avoid reloading on each request)
_rag_chain = None
_retriever = None

def initialize_rag():
    """Initialize the RAG system. Call this once at startup."""
    global _rag_chain, _retriever
    if _rag_chain is None:
        print("Initializing RAG system...")
        _rag_chain, _retriever = setup_rag_hybrid()
        print("RAG system initialized!")
    return _rag_chain, _retriever

def query_rag(question: str) -> dict:
    """
    Simple function to query the RAG system and return response.
    Perfect for FastAPI endpoints.

    Args:
        question: User's question

    Returns:
        dict with 'answer' and 'sources' keys
    """
    global _rag_chain, _retriever

    # Initialize if not already done
    if _rag_chain is None:
        initialize_rag()

    # Get answer from RAG chain
    answer = _rag_chain.invoke(question)

    # Get source documents
    source_docs = _retriever.invoke(question)

    # Format sources
    sources = []
    for doc in source_docs:
        sources.append({
            "page": doc.metadata.get('page', 'N/A'),
            "file": doc.metadata.get('source', 'N/A').split('/')[-1],
            "content": doc.page_content[:200] + "..."
        })

    return {
        "answer": answer,
        "sources": sources,
        "num_sources": len(sources)
    }

def ask_question(rag_chain, retriever, question):
    """Ask a question and get answer with sources."""

    print(f"\nQuestion: {question}")
    print("=" * 60)

    # Get answer from RAG chain
    answer = rag_chain.invoke(question)

    # Display answer
    print(f"\nAnswer:\n{answer}")

    # Get source documents separately using retriever
    source_docs = retriever.invoke(question)

    # Display source documents
    print(f"\n{'='*60}")
    print(f"Source Documents (retrieved {len(source_docs)} documents):")
    print("=" * 60)

    for i, doc in enumerate(source_docs, 1):
        print(f"\n[Source {i}]")
        print(f"Page: {doc.metadata.get('page', 'N/A')}")
        print(f"File: {doc.metadata.get('source', 'N/A').split('/')[-1]}")
        print(f"Content: {doc.page_content[:200]}...")

    print("\n" + "=" * 60)

def interactive_mode(rag_chain, retriever):
    """Interactive mode for asking multiple questions."""

    print("\n" + "=" * 60)
    print("Interactive RAG Mode (Hybrid Search + MultiQuery + Reranking)")
    print("=" * 60)
    print("Type 'quit' or 'exit' to stop\n")

    while True:
        question = input("\nYour question: ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not question:
            continue

        ask_question(rag_chain, retriever, question)

def main():
    # Setup RAG system
    #rag_chain, retriever = setup_rag()  # Basic RAG
    #rag_chain, retriever = setup_rag_multiquery()  # MultiQuery RAG
    #rag_chain, retriever = setup_rag_with_reranking()  # MultiQuery + FlashRank Reranking
    rag_chain, retriever = setup_rag_hybrid()  # Hybrid Search + MultiQuery + FlashRank

    # Example questions
    print("Running example questions...\n")
  
    example_questions = [
        "What is machine learning?",
        "Explain neural networks",
        "What are the main algorithms discussed in the document?"
    ]

    for question in example_questions:
        ask_question(rag_chain, retriever, question)
        print("\n")

    # Start interactive mode
    interactive_mode(rag_chain, retriever)

if __name__ == "__main__":
    main()
