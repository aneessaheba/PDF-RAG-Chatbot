from setup_rag_hybrid import setup_rag_hybrid

print("Loading RAG system...")
rag_chain, retriever = setup_rag_hybrid()
print("✓ RAG system loaded!\n")

# Ask a question
question = "What is machine learning?"
print(f"Question: {question}\n")

# NEW: Let's see what documents were retrieved
print("Retrieving documents...")
documents = retriever.get_relevant_documents(question)

# NEW: Print how many documents we got
print(f"✓ Found {len(documents)} documents\n")

# NEW: Let's look at the first document
print("=" * 60)
print("FIRST DOCUMENT RETRIEVED:")
print("=" * 60)
print(documents[0].page_content[:300])  # First 300 characters
print("...\n")

# Now get the answer
answer = rag_chain.invoke(question)
print("=" * 60)
print("ANSWER:")
print("=" * 60)
print(answer)

# ============================================================
# METRIC 1: RECALL@K
# ============================================================
print("=" * 60)
print("EVALUATING RECALL@K")
print("=" * 60)

# Keywords that should appear in relevant documents
keywords = ["machine learning", "ML", "learning", "algorithm"]

# Check each document
relevant_count = 0
for i, doc in enumerate(documents):
    doc_text = doc.page_content.lower()  # Convert to lowercase
    
    # Check if ANY keyword appears in this document
    is_relevant = any(keyword.lower() in doc_text for keyword in keywords)
    
    if is_relevant:
        relevant_count += 1
        print(f"✓ Document {i+1}: RELEVANT")
    else:
        print(f"✗ Document {i+1}: NOT RELEVANT")

# Calculate Recall@4
recall = relevant_count / len(documents)
print(f"\n📊 Recall@4 = {relevant_count}/{len(documents)} = {recall:.2%}")