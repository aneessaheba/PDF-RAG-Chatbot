from setup_rag_hybrid import setup_rag_hybrid
import time

print("Loading RAG system...")
rag_chain, retriever = setup_rag_hybrid()
print("✓ RAG system loaded!\n")

# Our test question
question = "What is machine learning?"
print(f"Question: {question}\n")

# ============================================================
# METRIC 4: LATENCY
# ============================================================
print("=" * 60)
print("MEASURING LATENCY")
print("=" * 60)

# We'll measure 3 times to get an average
num_runs = 3
print(f"\nRunning {num_runs} times to get accurate average...\n")

retrieval_times = []
generation_times = []
total_times = []

for run in range(1, num_runs + 1):
    print(f"Run {run}/{num_runs}:")
    
    # Measure retrieval time
    start = time.time()
    documents = retriever.get_relevant_documents(question)
    retrieval_time = time.time() - start
    retrieval_times.append(retrieval_time)
    
    # Measure total time (retrieval + generation)
    start = time.time()
    answer = rag_chain.invoke(question)
    total_time = time.time() - start
    total_times.append(total_time)
    
    # Calculate generation time
    generation_time = total_time - retrieval_time
    generation_times.append(generation_time)
    
    print(f"  Retrieval: {retrieval_time:.3f}s")
    print(f"  Generation: {generation_time:.3f}s")
    print(f"  Total: {total_time:.3f}s\n")

# Calculate averages
avg_retrieval = sum(retrieval_times) / num_runs
avg_generation = sum(generation_times) / num_runs
avg_total = sum(total_times) / num_runs

print("=" * 60)
print("AVERAGE LATENCY RESULTS")
print("=" * 60)
print(f"⏱️  Average Retrieval Time:  {avg_retrieval:.3f}s")
print(f"⏱️  Average Generation Time: {avg_generation:.3f}s")
print(f"⏱️  Average Total Time:      {avg_total:.3f}s")

# Interpret the results
print("\n" + "=" * 60)
print("PRODUCTION READINESS")
print("=" * 60)

if avg_total < 1:
    status = "⚡ EXCELLENT"
    comment = "Real-time performance! Production ready!"
elif avg_total < 3:
    status = "✅ GOOD"
    comment = "Acceptable for most applications"
elif avg_total < 5:
    status = "⚠️  FAIR"
    comment = "Noticeable delay. Consider optimization"
elif avg_total < 10:
    status = "😐 SLOW"
    comment = "Users will get impatient. Needs optimization"
else:
    status = "❌ TOO SLOW"
    comment = "Unacceptable for production. Major optimization needed"

print(f"\nStatus: {status}")
print(f"Comment: {comment}")

# Identify bottleneck
print("\n" + "=" * 60)
print("BOTTLENECK ANALYSIS")
print("=" * 60)

retrieval_percentage = (avg_retrieval / avg_total) * 100
generation_percentage = (avg_generation / avg_total) * 100

print(f"Retrieval: {retrieval_percentage:.1f}% of total time")
print(f"Generation: {generation_percentage:.1f}% of total time")

if retrieval_percentage > 60:
    print("\n🔍 Bottleneck: RETRIEVAL")
    print("Suggestions:")
    print("  - Reduce number of documents retrieved (lower k)")
    print("  - Remove MultiQuery (generates multiple searches)")
    print("  - Use faster embedding model")
    print("  - Cache frequent queries")
elif generation_percentage > 60:
    print("\n🤖 Bottleneck: GENERATION")
    print("Suggestions:")
    print("  - Use a faster LLM model")
    print("  - Reduce context size sent to LLM")
    print("  - Lower temperature parameter")
else:
    print("\n⚖️  Balanced: Both retrieval and generation take similar time")