"""
Recall Metric: Measures how many of the relevant documents were actually retrieved.
Higher score = More complete retrieval of relevant information.
"""
import time
import json
from pathlib import Path
from setup_graph_rag import setup_graph_rag
from setup_rag_hybrid import setup_rag_hybrid
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

def evaluate_recall(question, retrieved_docs, all_docs, llm, k=10):
    """
    Evaluate recall by checking how many relevant docs were retrieved.
    
    Steps:
    1. Get top-k most relevant docs from full corpus (ground truth)
    2. Check how many of those were in the retrieved set
    3. Recall = (Retrieved ∩ Relevant) / Relevant
    """
    # Get IDs of retrieved docs
    if isinstance(retrieved_docs[0], dict):
        retrieved_ids = set([doc['id'] for doc in retrieved_docs])
    else:
        # For hybrid RAG, we need to match by content
        retrieved_contents = set([doc.page_content for doc in retrieved_docs])
    
    # For ground truth, we'll use semantic similarity to find truly relevant docs
    # This is a proxy - ideally you'd have human labels
    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
    
    # Get query embedding
    query_embedding = embeddings.embed_query(question)
    
    # Simple relevance: check if key terms from question appear in doc
    question_terms = set(question.lower().split())
    
    relevant_count = 0
    retrieved_relevant_count = 0
    
    for i, doc_content in enumerate(all_docs):
        doc_terms = set(doc_content.lower().split())
        overlap = len(question_terms.intersection(doc_terms))
        
        # Consider doc relevant if it has significant term overlap
        if overlap >= 2:
            relevant_count += 1
            
            # Check if this relevant doc was retrieved
            if isinstance(retrieved_docs[0], dict):
                if i in retrieved_ids:
                    retrieved_relevant_count += 1
            else:
                if doc_content in retrieved_contents:
                    retrieved_relevant_count += 1
    
    recall = retrieved_relevant_count / relevant_count if relevant_count > 0 else 0
    
    return {
        "recall": recall,
        "retrieved_relevant": retrieved_relevant_count,
        "total_relevant": relevant_count,
        "retrieved_total": len(retrieved_docs)
    }

def generate_test_questions():
    """Generate test questions about machine learning."""
    return [
        "What is supervised learning?",
        "Explain the bias-variance tradeoff",
        "What is overfitting and how to prevent it?",
        "What is the difference between classification and regression?",
        "Explain gradient descent",
        "What is cross-validation?",
        "What is regularization?",
        "Explain the concept of loss function",
        "What is the purpose of train-test split?",
        "What are decision trees?",
    ]

def run_recall_evaluation():
    """Run recall evaluation for both RAG systems."""
    print("=" * 80)
    print("RECALL EVALUATION")
    print("=" * 80)
    print("\nRecall measures: What fraction of relevant docs were retrieved?")
    print()
    
    # Initialize evaluation LLM
    eval_llm = Ollama(model="llama3.2:3b", temperature=0)
    
    # Load all documents for ground truth
    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    all_docs_data = vectorstore.get()
    all_docs = all_docs_data['documents']
    
    print(f"Loaded {len(all_docs)} documents from corpus")
    
    # Setup systems
    print("\n[1/3] Setting up GraphRAG...")
    graph_rag_chain, G = setup_graph_rag()
    
    print("\n[2/3] Setting up Hybrid RAG...")
    hybrid_rag_chain, hybrid_retriever = setup_rag_hybrid()
    
    print("\n[3/3] Loading test questions...")
    questions = generate_test_questions()
    print(f"Loaded {len(questions)} test questions\n")
    
    # Results storage
    results = {
        "graphrag": [],
        "hybrid_rag": []
    }
    
    print("=" * 80)
    print("RUNNING EVALUATION")
    print("=" * 80)
    
    for i, question in enumerate(questions, 1):
        print(f"\n[Question {i}/{len(questions)}]")
        print(f"Q: {question}")
        print("-" * 80)
        
        # Evaluate GraphRAG
        print("\n[GraphRAG]")
        try:
            answer, docs = graph_rag_chain(question)
            recall_metrics = evaluate_recall(question, docs, all_docs, eval_llm)
            
            results["graphrag"].append({
                "question": question,
                "answer": answer,
                **recall_metrics
            })
            
            print(f"Retrieved: {recall_metrics['retrieved_total']} docs")
            print(f"Relevant in corpus: {recall_metrics['total_relevant']} docs")
            print(f"Retrieved relevant: {recall_metrics['retrieved_relevant']} docs")
            print(f"Recall: {recall_metrics['recall']:.2%}")
            
        except Exception as e:
            print(f"ERROR: {e}")
            results["graphrag"].append({"question": question, "error": str(e)})
        
        # Evaluate Hybrid RAG
        print("\n[Hybrid RAG]")
        try:
            answer = hybrid_rag_chain.invoke(question)
            docs = hybrid_retriever.invoke(question)
            recall_metrics = evaluate_recall(question, docs, all_docs, eval_llm)
            
            results["hybrid_rag"].append({
                "question": question,
                "answer": answer,
                **recall_metrics
            })
            
            print(f"Retrieved: {recall_metrics['retrieved_total']} docs")
            print(f"Relevant in corpus: {recall_metrics['total_relevant']} docs")
            print(f"Retrieved relevant: {recall_metrics['retrieved_relevant']} docs")
            print(f"Recall: {recall_metrics['recall']:.2%}")
            
        except Exception as e:
            print(f"ERROR: {e}")
            results["hybrid_rag"].append({"question": question, "error": str(e)})
    
    # Calculate statistics
    print("\n" + "=" * 80)
    print("RECALL RESULTS")
    print("=" * 80)
    
    for system_name, system_results in results.items():
        print(f"\n{system_name.upper()}:")
        print("-" * 40)
        
        valid_results = [r for r in system_results if "error" not in r]
        
        if valid_results:
            recalls = [r["recall"] for r in valid_results]
            avg_recall = sum(recalls) / len(recalls)
            min_recall = min(recalls)
            max_recall = max(recalls)
            
            avg_retrieved = sum(r["retrieved_total"] for r in valid_results) / len(valid_results)
            avg_relevant = sum(r["total_relevant"] for r in valid_results) / len(valid_results)
            
            print(f"Average Recall: {avg_recall:.2%}")
            print(f"Min Recall: {min_recall:.2%}")
            print(f"Max Recall: {max_recall:.2%}")
            print(f"Avg Docs Retrieved: {avg_retrieved:.1f}")
            print(f"Avg Relevant Docs: {avg_relevant:.1f}")
            print(f"Success Rate: {len(valid_results)}/{len(system_results)}")
    
    # Save results
    output_dir = Path("./evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "recall_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nResults saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    results = run_recall_evaluation()