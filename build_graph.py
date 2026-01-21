import networkx as nx
import pickle
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import json
from pathlib import Path
from tqdm import tqdm
import re

def extract_entities_simple(text):
    """
    Simple rule-based entity extraction (FAST - no LLM calls).
    Extracts capitalized terms and key ML concepts.
    """
    entities = set()
    
    # Common ML terms to look for (case-insensitive)
    ml_terms = [
        'supervised learning', 'unsupervised learning', 'reinforcement learning',
        'neural network', 'deep learning', 'machine learning',
        'gradient descent', 'backpropagation', 'overfitting', 'underfitting',
        'regularization', 'cross-validation', 'train test split',
        'precision', 'recall', 'accuracy', 'loss function',
        'classification', 'regression', 'clustering',
        'decision tree', 'random forest', 'svm', 'support vector machine',
        'linear regression', 'logistic regression',
        'bias', 'variance', 'hyperparameter',
        'feature engineering', 'dimensionality reduction',
        'pca', 'principal component analysis',
        'optimization', 'learning rate', 'epoch', 'batch',
        'activation function', 'sigmoid', 'relu', 'softmax',
        'convolutional', 'recurrent', 'lstm', 'gru',
        'transformer', 'attention', 'embedding'
    ]
    
    text_lower = text.lower()
    
    # Extract ML terms
    for term in ml_terms:
        if term in text_lower:
            entities.add(term.title())
    
    # Extract capitalized words (likely important terms)
    capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b', text)
    for term in capitalized:
        if len(term) > 3 and term not in ['The', 'This', 'That', 'These', 'Those']:
            entities.add(term)
    
    # Extract acronyms (2-5 capital letters)
    acronyms = re.findall(r'\b[A-Z]{2,5}\b', text)
    for acronym in acronyms:
        if acronym not in ['PDF', 'URL', 'HTML', 'HTTP', 'HTTPS']:
            entities.add(acronym)
    
    return list(entities)[:15]

def find_relationships(entities, text):
    """
    Simple co-occurrence based relationship detection (FAST).
    """
    relations = []
    text_lower = text.lower()
    
    present_entities = [e for e in entities if e.lower() in text_lower]
    
    for i, e1 in enumerate(present_entities):
        for e2 in present_entities[i+1:]:
            idx1 = text_lower.find(e1.lower())
            idx2 = text_lower.find(e2.lower())
            if abs(idx1 - idx2) < 200:
                relations.append({
                    'source': e1,
                    'target': e2,
                    'relation': 'related_to'
                })
    
    return relations

def build_knowledge_graph_fast(sample_size=None):
    """Build knowledge graph using rule-based extraction."""
    print("Building Knowledge Graph (FAST MODE - Rule-based extraction)...")
    
    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
    
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    
    total_docs = vectorstore._collection.count()
    print(f"Loaded {total_docs} documents from ChromaDB")
    
    all_docs = vectorstore.get()
    documents = all_docs['documents']
    metadatas = all_docs['metadatas']
    
    if sample_size and sample_size < len(documents):
        print(f"SAMPLING: Using only {sample_size} documents")
        import random
        indices = random.sample(range(len(documents)), sample_size)
        documents = [documents[i] for i in indices]
        metadatas = [metadatas[i] for i in indices]
    
    G = nx.DiGraph()
    doc_entities = {}
    
    print(f"Extracting entities and relationships from {len(documents)} documents...")
    print("Using FAST rule-based extraction (no LLM calls)")
    
    for i, (doc, meta) in enumerate(tqdm(zip(documents, metadatas), total=len(documents))):
        entities = extract_entities_simple(doc)
        relations = find_relationships(entities, doc)
        
        doc_entities[i] = entities
        
        for entity in entities:
            if not G.has_node(entity):
                G.add_node(entity, text=entity, doc_ids=[i])
            else:
                if i not in G.nodes[entity]['doc_ids']:
                    G.nodes[entity]['doc_ids'].append(i)
        
        for rel in relations:
            source = rel['source']
            target = rel['target']
            relation_type = rel['relation']
            
            if source and target:
                if G.has_edge(source, target):
                    G[source][target]['weight'] += 1
                else:
                    G.add_edge(source, target, relation=relation_type, weight=1)
    
    output_dir = Path("./graph_data")
    output_dir.mkdir(exist_ok=True)
    
    # Save using pickle
    with open(output_dir / "knowledge_graph.pkl", "wb") as f:
        pickle.dump(G, f)
    
    with open(output_dir / "doc_entities.json", "w") as f:
        json.dump(doc_entities, f, indent=2)
    
    print(f"\nGraph Statistics:")
    print(f"  Nodes (entities): {G.number_of_nodes()}")
    print(f"  Edges (relationships): {G.number_of_edges()}")
    if G.number_of_nodes() > 0:
        print(f"  Avg degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
    
    if G.number_of_nodes() > 0:
        sample_nodes = list(G.nodes())[:10]
        print(f"\nSample entities: {sample_nodes}")
    
    print("\nDetecting communities...")
    if G.number_of_edges() > 0:
        G_undirected = G.to_undirected()
        try:
            import community as community_louvain
            communities = community_louvain.best_partition(G_undirected)
            num_communities = len(set(communities.values()))
            print(f"  Found {num_communities} communities")
            
            with open(output_dir / "communities.json", "w") as f:
                json.dump(communities, f, indent=2)
        except ImportError:
            print("  Skipping community detection (install python-louvain)")
            communities = None
    else:
        print("  No edges to detect communities")
        communities = None
    
    print(f"\nGraph saved to: {output_dir}")
    
    return G, doc_entities, communities

if __name__ == "__main__":
    import sys
    
    sample_size = None
    if len(sys.argv) > 1:
        try:
            sample_size = int(sys.argv[1])
            print(f"Will sample {sample_size} documents")
        except ValueError:
            print("Usage: python build_graph.py [sample_size]")
    
    G, doc_entities, communities = build_knowledge_graph_fast(sample_size=sample_size)
    print("\n✅ Knowledge graph built successfully!")
    print("\nNext steps:")
    print("  1. Run evaluations: python latency.py")
    print("  2. Or run all: python run_all_metrics.py")