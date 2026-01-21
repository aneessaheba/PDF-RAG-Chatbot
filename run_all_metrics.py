"""
Master script to run all evaluation metrics and generate comparison report.
"""
import json
from pathlib import Path
from datetime import datetime

def run_all_evaluations():
    """Run all evaluation metrics sequentially."""
    print("=" * 80)
    print("RUNNING ALL EVALUATION METRICS")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Import here to avoid loading models until needed
    from faithfulness import run_faithfulness_evaluation
    from relevance import run_relevance_evaluation
    from recall import run_recall_evaluation
    from latency import run_latency_evaluation
    
    results_summary = {}
    
    # 1. Faithfulness
    print("\n" + "=" * 80)
    print("METRIC 1/4: FAITHFULNESS")
    print("=" * 80)
    try:
        faithfulness_results = run_faithfulness_evaluation()
        results_summary['faithfulness'] = "✓ Completed"
    except Exception as e:
        print(f"ERROR in faithfulness evaluation: {e}")
        results_summary['faithfulness'] = f"✗ Failed: {e}"
    
    # 2. Relevance
    print("\n\n" + "=" * 80)
    print("METRIC 2/4: RELEVANCE")
    print("=" * 80)
    try:
        relevance_results = run_relevance_evaluation()
        results_summary['relevance'] = "✓ Completed"
    except Exception as e:
        print(f"ERROR in relevance evaluation: {e}")
        results_summary['relevance'] = f"✗ Failed: {e}"
    
    # 3. Recall
    print("\n\n" + "=" * 80)
    print("METRIC 3/4: RECALL")
    print("=" * 80)
    try:
        recall_results = run_recall_evaluation()
        results_summary['recall'] = "✓ Completed"
    except Exception as e:
        print(f"ERROR in recall evaluation: {e}")
        results_summary['recall'] = f"✗ Failed: {e}"
    
    # 4. Latency
    print("\n\n" + "=" * 80)
    print("METRIC 4/4: LATENCY")
    print("=" * 80)
    try:
        latency_results = run_latency_evaluation(num_runs=3)
        results_summary['latency'] = "✓ Completed"
    except Exception as e:
        print(f"ERROR in latency evaluation: {e}")
        results_summary['latency'] = f"✗ Failed: {e}"
    
    # Generate summary report
    generate_summary_report(results_summary)
    
    print("\n" + "=" * 80)
    print("ALL EVALUATIONS COMPLETED")
    print("=" * 80)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nResults saved in: ./evaluation_results/")
    print("  - faithfulness_results.json")
    print("  - relevance_results.json")
    print("  - recall_results.json")
    print("  - latency_results.json")
    print("  - summary_report.txt")
    print("  - summary_report.json")

def generate_summary_report(results_summary):
    """Generate a summary report comparing both systems across all metrics."""
    output_dir = Path("./evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    # Load all results
    metrics_data = {}
    
    for metric in ['faithfulness', 'relevance', 'recall', 'latency']:
        file_path = output_dir / f"{metric}_results.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                metrics_data[metric] = json.load(f)
    
    # Calculate averages for each metric
    summary = {
        "graphrag": {},
        "hybrid_rag": {},
        "comparison": {}
    }
    
    for system in ["graphrag", "hybrid_rag"]:
        # Faithfulness
        if 'faithfulness' in metrics_data:
            faith_results = [r for r in metrics_data['faithfulness'][system] if 'faithfulness_score' in r]
            if faith_results:
                summary[system]['faithfulness'] = sum(r['faithfulness_score'] for r in faith_results) / len(faith_results)
        
        # Relevance
        if 'relevance' in metrics_data:
            rel_results = [r for r in metrics_data['relevance'][system] if 'relevance_score' in r]
            if rel_results:
                summary[system]['relevance'] = sum(r['relevance_score'] for r in rel_results) / len(rel_results)
        
        # Recall
        if 'recall' in metrics_data:
            recall_results = [r for r in metrics_data['recall'][system] if 'recall' in r]
            if recall_results:
                summary[system]['recall'] = sum(r['recall'] for r in recall_results) / len(recall_results)
        
        # Latency
        if 'latency' in metrics_data:
            lat_results = [r for r in metrics_data['latency'][system] if 'avg_total_time' in r]
            if lat_results:
                summary[system]['latency'] = sum(r['avg_total_time'] for r in lat_results) / len(lat_results)
    
    # Generate text report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("EVALUATION SUMMARY REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Overall comparison table
    report_lines.append("OVERALL COMPARISON")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Metric':<20} {'GraphRAG':<20} {'Hybrid RAG':<20} {'Winner':<20}")
    report_lines.append("-" * 80)
    
    # Faithfulness
    if 'faithfulness' in summary['graphrag'] and 'faithfulness' in summary['hybrid_rag']:
        g_faith = summary['graphrag']['faithfulness']
        h_faith = summary['hybrid_rag']['faithfulness']
        winner = "GraphRAG" if g_faith > h_faith else "Hybrid RAG" if h_faith > g_faith else "Tie"
        report_lines.append(f"{'Faithfulness (1-5)':<20} {g_faith:<20.2f} {h_faith:<20.2f} {winner:<20}")
    
    # Relevance
    if 'relevance' in summary['graphrag'] and 'relevance' in summary['hybrid_rag']:
        g_rel = summary['graphrag']['relevance']
        h_rel = summary['hybrid_rag']['relevance']
        winner = "GraphRAG" if g_rel > h_rel else "Hybrid RAG" if h_rel > g_rel else "Tie"
        report_lines.append(f"{'Relevance (1-5)':<20} {g_rel:<20.2f} {h_rel:<20.2f} {winner:<20}")
    
    # Recall
    if 'recall' in summary['graphrag'] and 'recall' in summary['hybrid_rag']:
        g_recall = summary['graphrag']['recall']
        h_recall = summary['hybrid_rag']['recall']
        winner = "GraphRAG" if g_recall > h_recall else "Hybrid RAG" if h_recall > g_recall else "Tie"
        report_lines.append(f"{'Recall (0-1)':<20} {g_recall:<20.2%} {h_recall:<20.2%} {winner:<20}")
    
    # Latency (lower is better)
    if 'latency' in summary['graphrag'] and 'latency' in summary['hybrid_rag']:
        g_lat = summary['graphrag']['latency']
        h_lat = summary['hybrid_rag']['latency']
        winner = "GraphRAG" if g_lat < h_lat else "Hybrid RAG" if h_lat < g_lat else "Tie"
        report_lines.append(f"{'Latency (seconds)':<20} {g_lat:<20.3f} {h_lat:<20.3f} {winner:<20}")
    
    report_lines.append("-" * 80)
    report_lines.append("")
    
    # Execution status
    report_lines.append("EXECUTION STATUS")
    report_lines.append("-" * 80)
    for metric, status in results_summary.items():
        report_lines.append(f"{metric.capitalize():<20} {status}")
    report_lines.append("")
    
    # Save text report
    text_report = "\n".join(report_lines)
    with open(output_dir / "summary_report.txt", "w") as f:
        f.write(text_report)
    
    # Save JSON report
    summary['execution_status'] = results_summary
    summary['generated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(output_dir / "summary_report.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print to console
    print("\n" + text_report)

if __name__ == "__main__":
    run_all_evaluations()