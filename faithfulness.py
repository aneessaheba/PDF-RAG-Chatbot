from setup_rag_hybrid import setup_rag_hybrid
import os
import google.generativeai as genai

# ------------------------------------------------------------------
# Gemini setup
# ------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Set GEMINI_API_KEY environment variable")

genai.configure(api_key=GEMINI_API_KEY)
eval_model = genai.GenerativeModel("gemini-3-pro-preview")

# ------------------------------------------------------------------
# Load RAG
# ------------------------------------------------------------------
print("Loading RAG system...")
rag_chain, retriever = setup_rag_hybrid()
print("✓ RAG system loaded!\n")

question = "What is machine learning?"
print("Question:", question)

# ------------------------------------------------------------------
# Get answer + documents
# ------------------------------------------------------------------
answer = rag_chain.invoke(question)
documents = retriever.invoke(question)

context = "\n\n".join(doc.page_content for doc in documents)

print("\n" + "=" * 60)
print("ANSWER")
print("=" * 60)
print(answer)

print("\n" + "=" * 60)
print("CONTEXT SNIPPET (used for fact-checking)")
print("=" * 60)
print(context[:800])  # debug visibility

# ------------------------------------------------------------------
# FAITHFULNESS
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("FAITHFULNESS EVALUATION")
print("=" * 60)

# Step 1: Extract claims
claim_prompt = f"""
Break the answer below into individual factual claims.
List each claim on a new numbered line.

Answer:
{answer}

Claims:
"""

response = eval_model.generate_content(claim_prompt)
claims_text = response.text.strip()

claims = [
    line.strip()
    for line in claims_text.splitlines()
    if line.strip() and line.strip()[0].isdigit()
]

print(f"\nExtracted {len(claims)} claims:\n")
for c in claims:
    print("•", c)

# Step 2: Verify claims
supported = 0

print("\n" + "-" * 60)
print("VERIFYING CLAIMS")
print("-" * 60)

for i, claim in enumerate(claims, 1):
    verify_prompt = f"""
You are a strict fact-checker.

CONTEXT:
{context[:6000]}

CLAIM:
{claim}

If the claim is supported, answer EXACTLY:
YES | "<short exact quote from context>"

If not supported, answer EXACTLY:
NO | NOT FOUND

Return one line only.
"""

    response = eval_model.generate_content(verify_prompt)
    result = response.text.strip().splitlines()[0]

    if result.upper().startswith("YES"):
        supported += 1
        print(f"\n✓ Claim {i}: SUPPORTED")
        print("  Evidence:", result)
    else:
        print(f"\n✗ Claim {i}: NOT SUPPORTED")
        print("  Evidence:", result)

# ------------------------------------------------------------------
# Faithfulness score
# ------------------------------------------------------------------
faithfulness = supported / len(claims) if claims else 1.0

print("\n" + "=" * 60)
print(f"FAITHFULNESS SCORE: {supported}/{len(claims)} = {faithfulness:.2%}")
print("=" * 60)

if faithfulness >= 0.8:
    print("✓ Low hallucination (good grounding)")
elif faithfulness >= 0.6:
    print("⚠ Moderate hallucination risk")
else:
    print("✗ High hallucination risk")


