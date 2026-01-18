from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

def test_vectordb():
    """Test the ChromaDB vector database with various operations."""

    print("Loading ChromaDB vector store...")

    # Initialize embeddings (same model used for indexing)
    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")

    # Load the existing ChromaDB
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    # Get collection info
    collection = vectorstore._collection
    print(f"\n{'='*60}")
    print(f"Database Statistics:")
    print(f"{'='*60}")
    print(f"Total entries: {collection.count()}")
    print(f"Collection name: {collection.name}")

    # Sample a few entries
    print(f"\n{'='*60}")
    print(f"Sample Entries (first 3):")
    print(f"{'='*60}")
    results = collection.get(limit=3, include=['documents', 'metadatas'])
    for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas']), 1):
        print(f"\n[Entry {i}]")
        print(f"Source: {metadata.get('source', 'N/A')}")
        print(f"Page: {metadata.get('page', 'N/A')}")
        print(f"Content preview: {doc[:200]}...")

    # Test similarity search
    print(f"\n{'='*60}")
    print(f"Similarity Search Test:")
    print(f"{'='*60}")

    test_queries = [
        "machine learning",
        "neural networks",
        "algorithms"
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 40)
        results = vectorstore.similarity_search(query, k=3)
        for i, doc in enumerate(results, 1):
            print(f"\n  Result {i}:")
            print(f"  Source: {doc.metadata.get('source', 'N/A')}")
            print(f"  Page: {doc.metadata.get('page', 'N/A')}")
            print(f"  Content: {doc.page_content[:150]}...")

    # Test similarity search with scores
    print(f"\n{'='*60}")
    print(f"Similarity Search with Scores:")
    print(f"{'='*60}")

    query = "What is machine learning?"
    print(f"\nQuery: '{query}'")
    print("-" * 40)
    results_with_scores = vectorstore.similarity_search_with_score(query, k=5)

    for i, (doc, score) in enumerate(results_with_scores, 1):
        print(f"\n  Result {i} (Score: {score:.4f}):")
        print(f"  Source: {doc.metadata.get('source', 'N/A')}")
        print(f"  Page: {doc.metadata.get('page', 'N/A')}")
        print(f"  Content: {doc.page_content[:150]}...")

    # Metadata Filtering Examples
    print(f"\n{'='*60}")
    print(f"Metadata Filtering Examples:")
    print(f"{'='*60}")

    # Get all unique pages and sources for filtering examples
    all_results = collection.get(include=['metadatas'])
    unique_pages = set(m.get('page') for m in all_results['metadatas'] if m.get('page') is not None)
    unique_sources = set(m.get('source') for m in all_results['metadatas'] if m.get('source'))

    print(f"\nAvailable pages: {sorted(unique_pages)[:10]}... (showing first 10)")
    print(f"Available sources: {list(unique_sources)}")

    # 1. Filter by specific page
    print(f"\n{'='*60}")
    print(f"Filter: Show all entries from page 0")
    print(f"{'='*60}")

    page_filter = collection.get(
        where={"page": 0},
        include=['documents', 'metadatas'],
        limit=3
    )
    print(f"Found {len(page_filter['documents'])} entries (showing 3)")
    for i, (doc, metadata) in enumerate(zip(page_filter['documents'], page_filter['metadatas']), 1):
        print(f"\n[Entry {i}]")
        print(f"Page: {metadata.get('page', 'N/A')}")
        print(f"Content preview: {doc[:150]}...")

    # 2. Filter by source (using the actual source path)
    if unique_sources:
        source_to_filter = list(unique_sources)[0]
        print(f"\n{'='*60}")
        print(f"Filter: Show entries from specific source")
        print(f"{'='*60}")
        print(f"Source: {source_to_filter}")

        source_filter = collection.get(
            where={"source": source_to_filter},
            include=['documents', 'metadatas'],
            limit=3
        )
        print(f"\nFound {len(source_filter['documents'])} entries (showing 3)")
        for i, (doc, metadata) in enumerate(zip(source_filter['documents'], source_filter['metadatas']), 1):
            print(f"\n[Entry {i}]")
            print(f"Page: {metadata.get('page', 'N/A')}")
            print(f"Content preview: {doc[:150]}...")

    # 3. Search within a specific page
    print(f"\n{'='*60}")
    print(f"Search within specific page (page 5):")
    print(f"{'='*60}")

    query = "learning"
    print(f"Query: '{query}' on page 5")
    print("-" * 40)

    page_search_results = vectorstore.similarity_search(
        query,
        k=3,
        filter={"page": 5}
    )

    if page_search_results:
        for i, doc in enumerate(page_search_results, 1):
            print(f"\n  Result {i}:")
            print(f"  Page: {doc.metadata.get('page', 'N/A')}")
            print(f"  Content: {doc.page_content[:150]}...")
    else:
        print("  No results found on page 5")

    # 4. Search within a specific source
    if unique_sources:
        print(f"\n{'='*60}")
        print(f"Search within specific source:")
        print(f"{'='*60}")

        query = "algorithm"
        source_to_search = list(unique_sources)[0]
        print(f"Query: '{query}'")
        print(f"Source filter: {source_to_search.split('/')[-1]}")
        print("-" * 40)

        source_search_results = vectorstore.similarity_search(
            query,
            k=3,
            filter={"source": source_to_search}
        )

        for i, doc in enumerate(source_search_results, 1):
            print(f"\n  Result {i}:")
            print(f"  Page: {doc.metadata.get('page', 'N/A')}")
            print(f"  Source: {doc.metadata.get('source', 'N/A').split('/')[-1]}")
            print(f"  Content: {doc.page_content[:150]}...")

    # 5. Multiple metadata filters (page range)
    print(f"\n{'='*60}")
    print(f"Advanced Filter: Search within page range (pages 0-2):")
    print(f"{'='*60}")

    query = "machine learning"
    print(f"Query: '{query}' on pages 0-2")
    print("-" * 40)

    # Search pages 0, 1, and 2
    range_results = []
    for page_num in [0, 1, 2]:
        results = vectorstore.similarity_search(
            query,
            k=2,
            filter={"page": page_num}
        )
        range_results.extend(results)

    if range_results:
        for i, doc in enumerate(range_results[:5], 1):  # Show top 5
            print(f"\n  Result {i}:")
            print(f"  Page: {doc.metadata.get('page', 'N/A')}")
            print(f"  Content: {doc.page_content[:150]}...")
    else:
        print("  No results found in specified page range")

    print(f"\n{'='*60}")
    print("Test completed successfully!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    test_vectordb()
