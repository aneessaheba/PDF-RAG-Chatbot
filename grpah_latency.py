"""
Latency Metric: Measures response time from query to answer.
Lower time = Faster system.
"""
import time
import json
from pathlib import Path
from setup_graph_rag import setup_graph_rag
from setup_rag_hybrid import setup_rag_hybrid
import statistics

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

def measure_latency(rag_chain, question, system_type="graphrag"):
    """Measure latency for a single query with detailed breakdown."""
    timings = {}
    
    # Total time
    start_total = time.time()
    
    try:
        if system_type == "graphrag":
            # GraphRAG has retrieval and generation
            start_retrieval = time.time()
            answer, docs = rag_chain(question)
            end_retrieval = time.time()
            
            timings['retrieval_time'] = end_retrieval - start_retrieval
            timings['generation_time'] = 0  # Included in retrieval for GraphRAG
            
        else:  # hybrid_rag
            start_retrieval = time.time()
            answer = rag_chain.invoke(question)
            end_retrieval = time.time()
            
            timings['retrieval_time'] = end_retrieval - start_retrieval
            timings['generation_time'] = 0  # Included in retrieval for Hybrid RAG
        
        end_total = time.time()
        timings['total_time'] = end_total - start_total
        timings['success'] = True
        timings['answer_length'] = len(answer)
        
    except Exception as e:
        timings['success'] = False
        timings['error'] = str(e)
        timings['total_time'] = time.time() - start_total
    
    return timings

def run_latency_evaluation(num_runs=3):
    """
    Run latency evaluation for both RAG systems.
    
    Args:
        num_runs: Number of times to run each query (for averaging)
    """
    print("=" * 80)
    print("LATENCY EVALUATION")
    print("=" * 80)
    print(f"\nLatency measures: How fast is each system? (averaged over {num_runs} runs)")
    print()
    
    # Setup systems
    print("[1/2] Setting up GraphRAG...")
    graph_rag_chain, G = setup_graph_rag()
    
    print("\n[2/2] Setting up Hybrid RAG...")
    hybrid_rag_chain, hybrid_retriever = setup_rag_hybrid()
    
    print("\n[3/3] Loading test questions...")
    questions = generate_test_questions()
    print(f"Loaded {len(questions)} test questions\n")
    
    # Results storage
    results = {
        "graphrag": [],
        "hybrid_rag": [],
        "metadata": {
            "num_runs_per_query": num_runs,
            "total_queries": len(questions)
        }
    }
    
    print("=" * 80)
    print("RUNNING EVALUATION")
    print("=" * 80)
    
    for i, question in enumerate(questions, 1):
        print(f"\n[Question {i}/{len(questions)}]")
        print(f"Q: {question}")
        print("-" * 80)
        
        # GraphRAG - Run multiple times and average
        print(f"\n[GraphRAG - {num_runs} runs]")
        graph_timings_all = []
        
        for run in range(num_runs):
            print(f"  Run {run+1}/{num_runs}...", end=" ")
            timings = measure_latency(graph_rag_chain, question, "graphrag")
            graph_timings_all.append(timings)
            if timings['success']:
                print(f"{timings['total_time']:.3f}s")
            else:
                print(f"FAILED")
        
        # Calculate averages for GraphRAG
        successful_runs = [t for t in graph_timings_all if t['success']]
        if successful_runs:
            avg_total = statistics.mean([t['total_time'] for t in successful_runs])
            std_total = statistics.stdev([t['total_time'] for t in successful_runs]) if len(successful_runs) > 1 else 0
            
            results["graphrag"].append({
                "question": question,
                "avg_total_time": avg_total,
                "std_total_time": std_total,
                "min_time": min(t['total_time'] for t in successful_runs),
                "max_time": max(t['total_time'] for t in successful_runs),
                "success_rate": len(successful_runs) / num_runs,
                "all_runs": graph_timings_all
            })
            
            print(f"  Average: {avg_total:.3f}s ± {std_total:.3f}s")
        else:
            results["graphrag"].append({
                "question": question,
                "error": "All runs failed"
            })
        
        # Hybrid RAG - Run multiple times and average
        print(f"\n[Hybrid RAG - {num_runs} runs]")
        hybrid_timings_all = []
        
        for run in range(num_runs):
            print(f"  Run {run+1}/{num_runs}...", end=" ")
            timings = measure_latency(hybrid_rag_chain, question, "hybrid")
            hybrid_timings_all.append(timings)
            if timings['success']:
                print(f"{timings['total_time']:.3f}s")
            else:
                print(f"FAILED")
        
        # Calculate averages for Hybrid RAG
        successful_runs = [t for t in hybrid_timings_all if t['success']]
        if successful_runs:
            avg_total = statistics.mean([t['total_time'] for t in successful_runs])
            std_total = statistics.stdev([t['total_time'] for t in successful_runs]) if len(successful_runs) > 1 else 0
            
            results["hybrid_rag"].append({
                "question": question,
                "avg_total_time": avg_total,
                "std_total_time": std_total,
                "min_time": min(t['total_time'] for t in successful_runs),
                "max_time": max(t['total_time'] for t in successful_runs),
                "success_rate": len(successful_runs) / num_runs,
                "all_runs": hybrid_timings_all
            })
            
            print(f"  Average: {avg_total:.3f}s ± {std_total:.3f}s")
        else:
            results["hybrid_rag"].append({
                "question": question,
                "error": "All runs failed"
            })
    
    # Calculate overall statistics
    print("\n" + "=" * 80)
    print("LATENCY RESULTS")
    print("=" * 80)
    
    for system_name, system_results in results.items():
        if system_name == "metadata":
            continue
            
        print(f"\n{system_name.upper()}:")
        print("-" * 40)
        
        valid_results = [r for r in system_results if "error" not in r]
        
        if valid_results:
            avg_times = [r["avg_total_time"] for r in valid_results]
            overall_avg = statistics.mean(avg_times)
            overall_std = statistics.stdev(avg_times) if len(avg_times) > 1 else 0
            overall_min = min(r["min_time"] for r in valid_results)
            overall_max = max(r["max_time"] for r in valid_results)
            
            print(f"Overall Average Latency: {overall_avg:.3f}s ± {overall_std:.3f}s")
            print(f"Fastest Response: {overall_min:.3f}s")
            print(f"Slowest Response: {overall_max:.3f}s")
            print(f"Success Rate: {len(valid_results)}/{len(system_results)}")
            
            # Latency distribution
            print("\nLatency Distribution:")
            for r in valid_results:
                bar_length = int(r["avg_total_time"] * 10)
                bar = "█" * bar_length
                print(f"  {r['question'][:40]:40} {bar} {r['avg_total_time']:.3f}s")
    
    # Compare systems
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    
    graph_valid = [r for r in results["graphrag"] if "error" not in r]
    hybrid_valid = [r for r in results["hybrid_rag"] if "error" not in r]
    
    if graph_valid and hybrid_valid:
        graph_avg = statistics.mean([r["avg_total_time"] for r in graph_valid])
        hybrid_avg = statistics.mean([r["avg_total_time"] for r in hybrid_valid])
        
        speedup = graph_avg / hybrid_avg
        
        print(f"\nGraphRAG avg: {graph_avg:.3f}s")
        print(f"Hybrid RAG avg: {hybrid_avg:.3f}s")
        
        if speedup > 1:
            print(f"Hybrid RAG is {speedup:.2f}x FASTER")
        else:
            print(f"GraphRAG is {1/speedup:.2f}x FASTER")
    
    # Save results
    output_dir = Path("./evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "latency_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nResults saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    # Run with 3 iterations per query for reliable averages
    results = run_latency_evaluation(num_runs=3)