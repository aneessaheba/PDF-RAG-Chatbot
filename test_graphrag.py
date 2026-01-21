"""
Simple test script for GraphRAG - Check if it can answer questions
"""
from setup_graph_rag import setup_graph_rag
import time

def test_graphrag():
    """Test GraphRAG with sample questions."""
    
    print("=" * 80)
    print("TESTING GRAPHRAG - Can it answer questions?")
    print("=" * 80)
    
    # Setup GraphRAG
    print("\n[1/2] Setting up GraphRAG system...")
    graph_rag_chain, G = setup_graph_rag()
    
    print(f"\nGraph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Test questions
    test_questions = [
        "What is supervised learning?",
        "Explain overfitting",
        "What is gradient descent?",
        "What is cross-validation?",
        "Explain neural networks"
    ]
    
    print("\n" + "=" * 80)
    print("RUNNING TESTS")
    print("=" * 80)
    
    results = []
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*80}")
        print(f"[Question {i}/{len(test_questions)}]")
        print(f"{'='*80}")
        print(f"Q: {question}")
        print("-" * 80)
        
        try:
            # Time the query
            start = time.time()
            answer, docs = graph_rag_chain(question)
            elapsed = time.time() - start
            
            print(f"\n✓ SUCCESS (took {elapsed:.2f}s)")
            print(f"\n📄 Retrieved {len(docs)} documents")
            
            # Show retrieved doc snippets
            if docs:
                print("\nRetrieved content preview:")
                for j, doc in enumerate(docs[:2], 1):
                    content_preview = doc['content'][:150].replace('\n', ' ')
                    print(f"  Doc {j}: {content_preview}...")
            else:
                print("\n⚠️  No documents retrieved!")
            
            print(f"\n💬 ANSWER:")
            print("-" * 80)
            print(answer)
            print("-" * 80)
            
            results.append({
                "question": question,
                "success": True,
                "answer": answer,
                "num_docs": len(docs),
                "time": elapsed
            })
            
        except Exception as e:
            print(f"\n✗ FAILED")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append({
                "question": question,
                "success": False,
                "error": str(e)
            })
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    successful = [r for r in results if r.get('success')]
    failed = [r for r in results if not r.get('success')]
    
    print(f"\n✓ Successful: {len(successful)}/{len(results)}")
    print(f"✗ Failed: {len(failed)}/{len(results)}")
    
    if successful:
        avg_time = sum(r['time'] for r in successful) / len(successful)
        avg_docs = sum(r['num_docs'] for r in successful) / len(successful)
        print(f"\nAverage response time: {avg_time:.2f}s")
        print(f"Average docs retrieved: {avg_docs:.1f}")
    
    if failed:
        print("\nFailed questions:")
        for r in failed:
            print(f"  - {r['question']}: {r.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 80)
    
    return results

if __name__ == "__main__":
    results = test_graphrag()
    
    print("\n✅ Test completed!")
    print("\nNext steps:")
    print("  - If successful: Run full metrics with run_all_metrics.py")
    print("  - If failed: Check the errors above")