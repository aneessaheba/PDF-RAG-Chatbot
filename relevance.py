from setup_rag_hybrid import setup_rag_hybrid
import os
import google.generativeai as genai
import re

# Setup Gemini
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)
eval_model = genai.GenerativeModel('gemini-3-pro-preview')

print("Loading RAG system...")
rag_chain, retriever = setup_rag_hybrid()
print("✓ RAG system loaded!\n")

# Ask a question
question = "What is machine learning?"
print(f"Question: {question}\n")

# Get the answer
print("Getting answer from RAG...")
answer = rag_chain.invoke(question)

print("=" * 60)
print("ANSWER:")
print("=" * 60)
print(answer)
print()

# ============================================================
# METRIC 3: ANSWER RELEVANCE
# ============================================================
print("=" * 60)
print("EVALUATING ANSWER RELEVANCE")
print("=" * 60)

# Ask Gemini to judge the answer
judge_prompt = f"""You are evaluating how well an answer addresses a question.

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

print("\n🤖 Asking Gemini to judge the answer...")
response = eval_model.generate_content(judge_prompt)
rating_text = response.text.strip()

print(f"Gemini's response: {rating_text}")

# Extract the number (handle cases where Gemini adds explanation)
rating = 3  # Default
numbers = re.findall(r'\b[1-5]\b', rating_text)
if numbers:
    rating = int(numbers[0])

print(f"\n📊 Rating: {rating}/5")

# Convert to percentage (0-100%)
relevance = (rating - 1) / 4  # Maps 1-5 to 0-1
percentage = relevance * 100

print(f"📊 Answer Relevance: {percentage:.1f}%")

# Interpretation
print("\n" + "=" * 60)
if relevance >= 0.8:
    print("✓ Excellent! Answer is on-topic and helpful")
elif relevance >= 0.6:
    print("⚠ Good, but could be more focused")
elif relevance >= 0.4:
    print("⚠ Partially relevant, misses some points")
else:
    print("✗ Poor relevance, doesn't address the question well")