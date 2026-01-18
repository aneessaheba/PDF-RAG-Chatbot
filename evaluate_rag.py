import time
import numpy as np
import re
import os
import google.generativeai as genai
from setup_rag_hybrid import setup_rag_hybrid

# Setup Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("⚠️  Set GEMINI_API_KEY environment variable!")
    exit(1)

genai.configure(api_key=GEMINI_API_KEY)
eval_model = genai.GenerativeModel('gemini-3-pro-preview')

# Test questions for "Basic ML for Beginners" PDF
test_questions = [
    {
        "question": "What is machine learning?",
        "ground_truth_answer": "Machine learning is a subset of AI that enables systems to learn from data without being explicitly programmed",
        "relevant_keywords": ["machine learning", "AI", "artificial intelligence", "data", "algorithm", "learn", "model"]
    },
    {
        "question": "What are the main types of machine learning?",
        "ground_truth_answer": "The main types are supervised learning, unsupervised learning, and reinforcement learning",
        "relevant_keywords": ["supervised", "unsupervised", "reinforcement", "learning", "types", "classification"]
    },
    {
        "question": "What is supervised learning?",
        "ground_truth_answer": "Supervised learning uses labeled data to train models to make predictions",
        "relevant_keywords": ["supervised", "labeled", "training", "data", "prediction", "target", "output"]
    },
    {
        "question": "How does a decision tree work?",
        "ground_truth_answer": "A decision tree splits data based on features to make predictions through a tree structure",
        "relevant_keywords": ["decision tree", "split", "node", "feature", "classification", "regression"]
    },
    {
        "question": "What is overfitting in machine learning?",
        "ground_truth_answer": "Overfitting occurs when a model learns training data too well and performs poorly on new data",
        "relevant_keywords": ["overfitting", "training", "generalization", "validation", "test", "performance"]
    },
    {
        "question": "What is the difference between classification and regression?",
        "ground_truth_answer": "Classification predicts categories while regression predicts continuous numerical values",
        "relevant_keywords": ["classification", "regression", "categorical", "continuous", "predict", "output"]
    },
    {
        "question": "What is a training set and test set?",
        "ground_truth_answer": "Training set is used to train the model, test set evaluates performance on unseen data",
        "relevant_keywords": ["training", "test", "validation", "dataset", "split", "evaluation"]
    },
    {
        "question": "What is cross-validation?",
        "ground_truth_answer": "Cross-validation is a technique to assess model performance by splitting data into multiple folds",
        "relevant_keywords": ["cross-validation", "fold", "k-fold", "validation", "evaluation", "performance"]
    },
]

print("=" * 80)
print("RAG EVALUATION STARTING")
print("=" * 80)

# Initialize RAG system
print("\n[Step 1/5] Initializing RAG system...")
print("-" * 80)
rag_chain, retriever = setup_rag_hybrid()
print("✓ RAG system ready!")

# Helper function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ==============================================================================
# METRIC 1: RECALL@K
# ==============================================================================

print("\n" + "=" * 80)
print("METRIC 1: RECALL@K")
print("=" * 80)

def calculate_recall_at_k(retrieved_docs, relevant_keywords, k=4):
    relevant_count = 0
    for doc in retrieved_docs[:k]:
        doc_text = doc.page_content.lower()
        if any(keyword.lower() in doc_text for keyword in relevant_keywords):
            relevant_count += 1
    recall = relevant_count / k if k > 0 else 0
    return recall, relevant_count

recall_scores = []
for i, test_case in enumerate(test_questions):
    print(f"\nQuestion {i+1}: {test_case['question']}")
    retrieved_docs = retriever.get_relevant_documents(test_case['question'])
    recall, relevant_found = calculate_recall_at_k(retrieved_docs, test_case['relevant_keywords'], k=4)
    recall_scores.append(recall)
    print(f"  → Retrieved {len(retrieved_docs)} documents")
    print(f"  → {relevant_found}/4 documents were relevant")
    print(f"  → Recall@4: {recall:.2%}")

avg_recall = np.mean(recall_scores)
print(f"\n{'='*40}")
print(f"Average Recall@4: {avg_recall:.2%}")
print(f"{'='*40}")

# ==============================================================================
# METRIC 2: FAITHFULNESS
# ==============================================================================

print("\n" + "=" * 80)
print("METRIC 2: FAITHFULNESS (Groundedness)")
print("=" * 80)

def calculate_faithfulness(answer, context):
    extraction_prompt = f"""Break down the following answer into individual factual claims or statements.
List each claim on a new line, numbered.

Answer: {answer}

Claims:"""
    
    try:
        response = eval_model.generate_content(extraction_prompt)
        claims_text = response.text
    except Exception as e:
        print(f"    ⚠️  Error calling Gemini API: {e}")
        return 1.0, 0, 0
    
    claims = [line.strip() for line in claims_text.split('\n') 
              if line.strip() and any(c.isdigit() for c in line[:3])]
    
    if not claims:
        return 1.0, 0, 0
    
    print(f"  → Extracted {len(claims)} claims to verify")
    
    supported_count = 0
    for claim in claims:
        verification_prompt = f"""Context: {context}

Claim: {claim}

Question: Can this claim be directly supported or inferred from the context above?
Answer only 'YES' or 'NO'.

Answer:"""
        
        try:
            response = eval_model.generate_content(verification_prompt)
            verification = response.text.strip().upper()
            if 'YES' in verification:
                supported_count += 1
        except Exception as e:
            print(f"    ⚠️  Error verifying claim: {e}")
            continue
    
    faithfulness = supported_count / len(claims) if len(claims) > 0 else 1.0
    return faithfulness, supported_count, len(claims)

faithfulness_scores = []
for i, test_case in enumerate(test_questions):
    print(f"\nQuestion {i+1}: {test_case['question']}")
    start = time.time()
    answer = rag_chain.invoke(test_case['question'])
    generation_time = time.time() - start
    retrieved_docs = retriever.get_relevant_documents(test_case['question'])
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    print(f"  → Generated answer in {generation_time:.2f}s")
    print(f"  → Answer: {answer[:100]}...")
    faithfulness, supported, total = calculate_faithfulness(answer, context)
    faithfulness_scores.append(faithfulness)
    print(f"  → {supported}/{total} claims supported")
    print(f"  → Faithfulness: {faithfulness:.2%}")

avg_faithfulness = np.mean(faithfulness_scores)
print(f"\n{'='*40}")
print(f"Average Faithfulness: {avg_faithfulness:.2%}")
print(f"{'='*40}")

# ==============================================================================
# METRIC 3: ANSWER RELEVANCE
# ==============================================================================

print("\n" + "=" * 80)
print("METRIC 3: ANSWER RELEVANCE")
print("=" * 80)

def calculate_answer_relevance(question, answer, ground_truth=None):
    judge_prompt = f"""You are evaluating the quality of an answer to a question.

Question: {question}

Answer: {answer}

Rate how well the answer addresses the question on a scale of 1-5:
5 = Perfectly answers the question
4 = Good answer, addresses main points
3 = Partially answers the question
2 = Somewhat related but misses key points
1 = Irrelevant or doesn't answer the question

Provide ONLY a number from 1-5.

Rating:"""
    
    try:
        response = eval_model.generate_content(judge_prompt)
        rating_text = response.text.strip()
    except Exception as e:
        print(f"    ⚠️  Error calling Gemini API: {e}")
        return 0.5, 3
    
    rating = 3
    numbers = re.findall(r'\b[1-5]\b', rating_text)
    if numbers:
        rating = int(numbers[0])
    
    relevance = (rating - 1) / 4
    return relevance, rating

relevance_scores = []
for i, test_case in enumerate(test_questions):
    print(f"\nQuestion {i+1}: {test_case['question']}")
    answer = rag_chain.invoke(test_case['question'])
    print(f"  → Answer: {answer[:100]}...")
    relevance, rating = calculate_answer_relevance(test_case['question'], answer, test_case['ground_truth_answer'])
    relevance_scores.append(relevance)
    print(f"  → Gemini Rating: {rating}/5")
    print(f"  → Relevance Score: {relevance:.2%}")

avg_relevance = np.mean(relevance_scores)
print(f"\n{'='*40}")
print(f"Average Answer Relevance: {avg_relevance:.2%}")
print(f"{'='*40}")

# ==============================================================================
# METRIC 4: LATENCY
# ==============================================================================

print("\n" + "=" * 80)
print("METRIC 4: LATENCY")
print("=" * 80)

def measure_latency(question, rag_chain, retriever, num_runs=3):
    retrieval_times = []
    generation_times = []
    total_times = []
    
    for _ in range(num_runs):
        start = time.time()
        docs = retriever.get_relevant_documents(question)
        retrieval_time = time.time() - start
        retrieval_times.append(retrieval_time)
        
        start = time.time()
        answer = rag_chain.invoke(question)
        total_time = time.time() - start
        total_times.append(total_time)
        
        generation_time = total_time - retrieval_time
        generation_times.append(generation_time)
    
    return np.mean(retrieval_times), np.mean(generation_times), np.mean(total_times)

print("\nMeasuring latency (3 runs per question)...\n")

all_retrieval = []
all_generation = []
all_total = []

for i, test_case in enumerate(test_questions):
    print(f"Question {i+1}: {test_case['question']}")
    ret_time, gen_time, total_time = measure_latency(test_case['question'], rag_chain, retriever, num_runs=3)
    all_retrieval.append(ret_time)
    all_generation.append(gen_time)
    all_total.append(total_time)
    print(f"  → Retrieval: {ret_time:.3f}s")
    print(f"  → Generation: {gen_time:.3f}s")
    print(f"  → Total: {total_time:.3f}s")
    
    if total_time < 1:
        status = "✓ Excellent"
    elif total_time < 3:
        status = "✓ Good"
    elif total_time < 5:
        status = "⚠ Fair"
    else:
        status = "✗ Poor"
    print(f"  → Status: {status}")

print(f"\n{'='*40}")
print(f"Average Retrieval Time: {np.mean(all_retrieval):.3f}s")
print(f"Average Generation Time: {np.mean(all_generation):.3f}s")
print(f"Average Total Time: {np.mean(all_total):.3f}s")
print(f"{'='*40}")

# ==============================================================================
# FINAL SUMMARY REPORT
# ==============================================================================

print("\n" + "=" * 80)
print("FINAL EVALUATION REPORT")
print("=" * 80)

print(f"""
┌─────────────────────────────────────────┐
│  METRIC                    SCORE         │
├─────────────────────────────────────────┤
│  Recall@4                  {avg_recall:6.2%}       │
│  Faithfulness              {avg_faithfulness:6.2%}       │
│  Answer Relevance          {avg_relevance:6.2%}       │
│  Avg Latency               {np.mean(all_total):6.3f}s      │
└─────────────────────────────────────────┘

INTERPRETATION:
---------------
Recall@4: {avg_recall:.0%} of retrieved docs were relevant
  {'✓ Great!' if avg_recall > 0.7 else '⚠ Consider improving retrieval' if avg_recall > 0.5 else '✗ Needs work'}

Faithfulness: {avg_faithfulness:.0%} of claims were grounded in context
  {'✓ Low hallucination!' if avg_faithfulness > 0.8 else '⚠ Some hallucinations' if avg_faithfulness > 0.6 else '✗ High hallucination risk'}

Answer Relevance: {avg_relevance:.0%} relevance to questions
  {'✓ Answers on-topic!' if avg_relevance > 0.8 else '⚠ Sometimes off-topic' if avg_relevance > 0.6 else '✗ Often misses the point'}

Latency: {np.mean(all_total):.2f}s average response time
  {'✓ Production ready!' if np.mean(all_total) < 3 else '⚠ Acceptable but could be faster' if np.mean(all_total) < 5 else '✗ Too slow for production'}
""")

print("\nNEXT STEPS:")
print("-" * 80)
if avg_recall < 0.7:
    print("• Improve retrieval: Try adjusting k, chunk size, or embedding model")
if avg_faithfulness < 0.8:
    print("• Reduce hallucinations: Add stricter prompts or use a better LLM")
if avg_relevance < 0.8:
    print("• Improve relevance: Better prompt engineering or question understanding")
if np.mean(all_total) > 3:
    print("• Optimize latency: Cache embeddings, use faster models, or async retrieval")

print("\n" + "=" * 80)
print("Evaluation complete!")
print("=" * 80)