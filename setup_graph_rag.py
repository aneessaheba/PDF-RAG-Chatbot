import networkx as nx
import pickle
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import numpy as np
from pathlib import Path

def load_graph():
    """Load the knowledge graph and related data."""
    graph_dir = Path("./graph_data")
    
    # Load graph using pickle
    with open(graph_dir / "knowledge_graph.pkl", "rb") as f:
        G = pickle.load(f)
    
    with open(graph_dir / "doc_entities.json", "r") as f:
        doc_entities = json.load(f)
    
    communities = None
    if (graph_dir / "communities.json").exists():
        with open(graph_dir / "communities.json", "r") as f:
            communities = json.load(f)
    
    return G, doc_entities, communities

def extract_query_entities(query, llm):
    """Extract key entities from the query."""
    prompt = ChatPromptTemplate.from_template("""
Extract the main concepts, terms, or entities from this question.
Return ONLY a JSON array of strings: ["entity1", "entity2", ...]

Question: {query}

JSON array:""")
    
    chain = prompt | llm
    try:
        response = chain.invoke({"query": query}).strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        
        entities = json.loads(response)
        return entities if isinstance(entities, list) else []
    except:
        # Fallback: extract important words
        return [word.lower() for word in query.split() if len(word) > 4]

def graph_retrieve(query, G, doc_entities, vectorstore, llm, k=4, hops=2):
    """
    Retrieve documents using graph-based approach:
    1. Extract entities from query
    2. Find matching nodes in graph
    3. Expand via graph traversal (k-hops)
    4. Retrieve documents containing expanded entities
    """
    # Extract query entities
    query_entities = extract_query_entities(query, llm)
    print(f"Query entities: {query_entities}")
    
    # Find matching nodes in graph (fuzzy match)
    matched_nodes = set()
    for entity in query_entities:
        entity_lower = entity.lower()
        for node in G.nodes():
            if entity_lower in node.lower() or node.lower() in entity_lower:
                matched_nodes.add(node)
    
    print(f"Matched graph nodes: {matched_nodes}")
    
    # Expand via graph traversal
    expanded_nodes = set(matched_nodes)
    for node in matched_nodes:
        # Get neighbors within k hops
        for neighbor in nx.single_source_shortest_path_length(G, node, cutoff=hops).keys():
            expanded_nodes.add(neighbor)
    
    print(f"Expanded to {len(expanded_nodes)} nodes after {hops}-hop traversal")
    
    # Get document IDs containing these entities
    relevant_doc_ids = set()
    for node in expanded_nodes:
        if G.has_node(node):
            doc_ids = G.nodes[node].get('doc_ids', [])
            relevant_doc_ids.update(doc_ids)
    
    print(f"Found {len(relevant_doc_ids)} relevant documents")
    
    # Retrieve documents from vectorstore
    all_docs_data = vectorstore.get()
    documents = all_docs_data['documents']
    metadatas = all_docs_data['metadatas']
    
    # Get the actual documents
    retrieved_docs = []
    for doc_id in relevant_doc_ids:
        if doc_id < len(documents):
            retrieved_docs.append({
                'content': documents[doc_id],
                'metadata': metadatas[doc_id],
                'id': doc_id
            })
    
    # Limit to top k (you could rank by centrality or other metrics)
    retrieved_docs = retrieved_docs[:k]
    
    return retrieved_docs

def format_docs(docs):
    """Format retrieved documents."""
    return "\n\n".join(doc['content'] for doc in docs)

def setup_graph_rag():
    """Setup GraphRAG retrieval system."""
    print("Setting up GraphRAG system...")
    
    # Load graph
    G, doc_entities, communities = load_graph()
    print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Initialize embeddings and vectorstore
    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    
    # Initialize LLM
    llm = Ollama(model="llama3.2:3b", temperature=0.7)
    
    # Create answer template
    template = """You are a helpful AI assistant. Use ONLY the context below to answer the question.
If the answer is in the context, provide it clearly and concisely.
If the answer is NOT in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    def graph_rag_chain(question):
        """GraphRAG pipeline."""
        # Retrieve using graph
        docs = graph_retrieve(question, G, doc_entities, vectorstore, llm, k=4, hops=2)
        context = format_docs(docs)
        
        # Generate answer
        answer_chain = prompt | llm | StrOutputParser()
        answer = answer_chain.invoke({"context": context, "question": question})
        
        return answer, docs
    
    print("\nGraphRAG system ready!")
    print("Pipeline:")
    print("  1. Extract entities from query")
    print("  2. Match entities to graph nodes")
    print("  3. Expand via k-hop graph traversal")
    print("  4. Retrieve documents containing expanded entities")
    print("  5. Generate answer\n")
    
    return graph_rag_chain, G

if __name__ == "__main__":
    graph_rag_chain, G = setup_graph_rag()
    
    # Test query
    test_query = "What is supervised learning?"
    print(f"Test query: {test_query}")
    answer, docs = graph_rag_chain(test_query)
    print(f"\nAnswer: {answer}")
    print(f"\nRetrieved {len(docs)} documents")