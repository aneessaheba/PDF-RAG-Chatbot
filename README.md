# PDF Chatbot RAG System - Complete Beginner's Guide

## Table of Contents
1. [Introduction](#introduction)
2. [What You'll Learn](#what-youll-learn)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Project Structure](#project-structure)
6. [Learning Path](#learning-path)
7. [Design Decisions Explained](#design-decisions-explained)
8. [Detailed Module Explanations](#detailed-module-explanations)
9. [Testing & Troubleshooting](#testing--troubleshooting)
10. [Common Terms Glossary](#common-terms-glossary)

---

## Introduction

### What is AI and Generative AI?

**Artificial Intelligence (AI)** is when computers perform tasks that typically require human intelligence, like understanding language, recognizing images, or making decisions.

**Generative AI (Gen AI)** is a type of AI that can create new content - like writing text, generating images, or answering questions. Think of ChatGPT or Claude - they're Generative AI models that can understand your questions and generate helpful responses.

### What is RAG (Retrieval-Augmented Generation)?

Imagine you're writing an essay about a topic. You could either:
1. Write from memory (what you already know)
2. Look up information in books first, then write based on what you found

**RAG works like option 2.** Instead of just relying on what the AI "remembers" from training, RAG:
1. **Retrieves** relevant information from your documents
2. **Augments** (adds) that information to the AI's context
3. **Generates** an answer based on your actual documents

This makes the AI much more accurate and prevents it from making things up.

### Why Build This System?

This project shows you how to build a chatbot that can answer questions about YOUR documents (like PDFs). By the end, you'll have:
- A system that understands your PDF documents
- Multiple ways to improve answer accuracy
- A web API that other applications can use

---

## What You'll Learn

This guide takes you from **zero to hero** in building RAG systems:

1. **Basic concepts**: What are embeddings, vector databases, and LLMs?
2. **Basic RAG**: Simple question-answering from PDFs
3. **MultiQuery RAG**: Generate multiple search queries for better results
4. **RAG with Reranking**: Sort results to find the most relevant ones
5. **Hybrid RAG**: Combine keyword and semantic search for best results
6. **API Development**: Make your chatbot accessible via web API
7. **Dynamic PDF Upload**: Add documents through API endpoints
8. **Web Front-End**: Build a user-friendly interface with HTML/CSS/JavaScript

---

## Prerequisites

### What You Need

1. **A Computer**: Windows, Mac, or Linux
2. **Python 3.9+**: Programming language we'll use
   - Check if installed: Open terminal/command prompt, type `python --version`
   - Download from: https://www.python.org/downloads/

3. **Ollama**: Runs AI models on your computer (free)
   - Download from: https://ollama.ai
   - This lets you run AI models locally without paying for cloud services

4. **A PDF file**: Any PDF document you want to ask questions about
   - The project includes "Foundations_of_Machine_Learning.pdf" as an example

5. **Basic Terminal/Command Prompt Skills**:
   - How to navigate folders with `cd` command
   - How to run Python scripts with `python script_name.py`

### No Prior Programming? No Problem!

This guide assumes you're completely new to:
- Programming
- AI/Machine Learning
- APIs
- Vector databases

We'll explain everything as we go.

---

## Installation

### Step 1: Install Ollama

1. Download Ollama from https://ollama.ai
2. Install it on your computer
3. Open terminal and pull the required models:

```bash
# Download the embedding model (converts text to numbers)
ollama pull qwen3-embedding:0.6b

# Download the language model (generates answers)
ollama pull llama3.2:1b
```

**What just happened?**
- `qwen3-embedding:0.6b`: A small model (600 million parameters) that converts text into mathematical vectors
- `llama3.2:1b`: A small language model (1 billion parameters) that generates human-like text

### Step 2: Install Python Dependencies

1. Open terminal/command prompt
2. Navigate to this project folder:
```bash
cd /path/to/PDF_Chatbot
```

3. Install required Python packages:
```bash
pip install -r requirements.txt
```

**What's in requirements.txt?**
- `langchain`: Framework for building AI applications
- `chromadb`: Vector database to store document embeddings
- `pypdf`: Read PDF files
- `flashrank`: Reranking model for better search results
- `rank_bm25`: Keyword-based search algorithm
- `fastapi`: Framework to build web APIs
- `uvicorn`: Server to run the API

---

## Project Structure

Here's what each file does:

```
PDF_Chatbot/
│
├── Foundations_of_Machine_Learning.pdf  # Sample PDF document
│
├── requirements.txt                      # List of Python packages needed
│
├── Indexer.py                           # STEP 0: Convert PDF to searchable format
│
├── setup_rag.py                         # STEP 1: Basic RAG system
│
├── setup_rag_multiquery.py              # STEP 2: RAG with multiple query variations
│
├── setup_rag_rerank.py                  # STEP 3: RAG with result reranking
│
├── setup_rag_hybrid.py                  # STEP 4: Hybrid search (best version)
│
├── ask_questions.py                     # STEP 5: Interactive Q&A interface
│
├── api.py                               # STEP 6: FastAPI web endpoint (query only)
│
├── api_enhanced.py                      # STEP 7: Enhanced API with PDF upload
│
├── index.html                           # STEP 8: Web front-end (HTML structure)
│
├── style.css                            # STEP 8: Web front-end (styling)
│
├── app.js                               # STEP 8: Web front-end (JavaScript logic)
│
├── test_vectordb.py                     # Test and explore the vector database
│
├── uploaded_pdfs/                       # Folder for uploaded PDFs (created automatically)
│
└── chroma_db/                           # Folder where embeddings are stored
```

---

## Learning Path

Follow these steps in order. Each step builds on the previous one.

### STEP 0: Index Your PDF (Creating the Knowledge Base)

**File**: `Indexer.py`

**What it does**: Converts your PDF into a format the AI can search through.

**Run it**:
```bash
python Indexer.py
```

**What happens behind the scenes**:

1. **Load PDF**: Reads the PDF file page by page
2. **Split into Chunks**: Breaks long text into smaller pieces (1000 characters each, with 200 character overlap)
   - Why chunks? AI models have limited memory. Smaller chunks are easier to search.
   - Why overlap? Ensures concepts spanning chunk boundaries aren't lost.
3. **Create Embeddings**: Each chunk is converted to a vector (list of numbers)
   - Example: "machine learning" → [0.2, -0.5, 0.8, ..., 0.3] (hundreds of dimensions)
   - Similar concepts have similar vectors
4. **Store in ChromaDB**: Saves vectors to disk for fast searching later

**Key Parameters**:
- `chunk_size=1000`: Each piece of text is ~1000 characters
- `chunk_overlap=200`: 200 characters overlap between chunks
- `model="qwen3-embedding:0.6b"`: The embedding model used

**Output**: Creates `chroma_db/` folder with indexed document chunks.

---

### STEP 1: Basic RAG System

**File**: `setup_rag.py`

**What it does**: The simplest RAG system - search documents and generate answers.

**How it works**:

```
Your Question → Search Vector DB → Find Top 4 Similar Chunks → Give to LLM → Get Answer
```

**Run it**:
```bash
python setup_rag.py
```

**Key Components**:

1. **Embeddings Model** (`OllamaEmbeddings`):
   - Converts your question to a vector
   - Uses same model as indexing (`qwen3-embedding:0.6b`)

2. **Vector Store** (`Chroma`):
   - Loads the indexed documents from `chroma_db/`
   - Performs similarity search

3. **Retriever**:
   - Finds top 4 most relevant chunks (`k=4`)
   - Uses cosine similarity to compare vectors

4. **LLM** (`Ollama`):
   - Generates natural language answers
   - Model: `llama3.2:1b`
   - Temperature: 0.7 (controls randomness - higher = more creative)

5. **Prompt Template**:
   - Instructions for the LLM on how to answer
   - Tells it to use only the provided context
   - Instructs it to say "I don't know" if context doesn't have the answer

6. **RAG Chain**:
   - Connects all components in a pipeline
   - Uses LangChain Expression Language (LCEL)

**Code Breakdown**:

```python
# This creates the complete pipeline
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

**What this means**:
1. Question goes to retriever → gets relevant documents
2. Documents are formatted into text
3. Question + Context go through the prompt template
4. LLM generates an answer
5. Output parser extracts the text

---

### STEP 2: MultiQuery RAG (Better Retrieval)

**File**: `setup_rag_multiquery.py`

**Problem it solves**: Sometimes one question phrasing doesn't find all relevant info.

**Example**:
- You ask: "What is ML?"
- Better to also search: "machine learning definition", "what does machine learning mean", "ML explanation"

**How it works**:

```
Your Question → LLM generates 3 variations → Search each variation →
Combine results → Remove duplicates → Give to LLM → Answer
```

**Run it**:
```bash
python setup_rag_multiquery.py
```

**What's different from Step 1**:

```python
multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
)
```

**What this does**:
1. Takes your question
2. Uses LLM to generate 2-3 alternative phrasings
3. Searches for each variation
4. Combines and de-duplicates results
5. Returns more comprehensive context

**Why it's better**: Catches relevant chunks that a single query might miss.

**Logging**:
- You'll see the generated query variations in the console
- Helpful for understanding what searches are being performed

---

### STEP 3: RAG with Reranking (Better Ranking)

**File**: `setup_rag_rerank.py`

**Problem it solves**: Vector search sometimes puts less relevant results in top positions.

**Solution**: Use a specialized "reranking" model to re-sort results by relevance.

**How it works**:

```
Question → MultiQuery (get variations) → Search (get 10 chunks) →
Rerank (sort by relevance) → Keep top 4 → Give to LLM → Answer
```

**Run it**:
```bash
python setup_rag_rerank.py
```

**What's different from Step 2**:

```python
# Get more results initially (10 instead of 4)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Add reranking layer
compressor = FlashrankRerank(
    model="ms-marco-MiniLM-L-12-v2",
    top_n=4  # Only keep top 4 after reranking
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=multiquery_retriever
)
```

**What this does**:
1. Retrieves 10 potentially relevant chunks
2. Reranker reads each chunk + your question
3. Scores each chunk for actual relevance
4. Returns only the top 4 most relevant

**Why it's better**:
- Vector search is fast but approximate
- Reranker is slower but more accurate
- Best of both: Fast retrieval + Accurate ranking

**The FlashRank Model**:
- Lightweight neural model trained on Microsoft Marco dataset
- Specifically trained to judge relevance
- Much faster than full transformer rerankers

---

### STEP 4: Hybrid RAG (Best Version)

**File**: `setup_rag_hybrid.py`

**Problem it solves**: Different search methods have different strengths:
- **Semantic search** (vectors): Good for concepts, synonyms, meaning
  - Example: Finds "ML" when you search "machine learning"
- **Keyword search** (BM25): Good for exact terms, names, acronyms
  - Example: Finds "TF-IDF" when you search "TF-IDF"

**Solution**: Combine both approaches!

**How it works**:

```
                    ┌─→ BM25 Search (keywords) ─┐
Your Question ──────┤                            ├─→ Ensemble → Rerank → Top 4 → LLM → Answer
                    └─→ MultiQuery + Semantic ──┘
```

**Run it**:
```bash
python setup_rag_hybrid.py
```

**Architecture Breakdown**:

```python
# 1. Semantic retriever (vector search)
semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

# 2. Add MultiQuery to semantic
multiquery_semantic = MultiQueryRetriever.from_llm(
    retriever=semantic_retriever,
    llm=llm
)

# 3. BM25 retriever (keyword search)
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 8

# 4. Combine both with equal weight
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, multiquery_semantic],
    weights=[0.5, 0.5]
)

# 5. Add reranking on top
final_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=hybrid_retriever
)
```

**What each component does**:

1. **BM25 Retriever**:
   - Classic information retrieval algorithm
   - Scores based on term frequency and document frequency
   - Great for exact matches and rare terms

2. **Semantic Retriever + MultiQuery**:
   - Understands meaning and context
   - Generates query variations
   - Finds conceptually related content

3. **Ensemble Retriever**:
   - Combines results from both retrievers
   - `weights=[0.5, 0.5]` means equal importance
   - Uses Reciprocal Rank Fusion to merge results

4. **FlashRank Reranker**:
   - Final quality check
   - Ensures only the best 4 chunks are used

**Why this is the best**:
- Catches results that either method alone would miss
- Robust to different query styles
- Highest accuracy for most use cases

**Trade-off**: Slightly slower because it runs two search methods, but still fast enough for real-time use.

---

### STEP 5: Interactive Q&A

**File**: `ask_questions.py`

**What it does**: Provides a user-friendly interface to ask questions.

**Run it**:
```bash
python ask_questions.py
```

**Features**:

1. **Example Questions**: Runs 3 pre-defined questions automatically
2. **Interactive Mode**: Lets you ask unlimited questions
3. **Source Display**: Shows which document chunks were used
4. **Configurable**: Easy to switch between RAG versions

**Functions Explained**:

```python
def query_rag(question: str) -> dict:
    """
    Simplified function for API use.
    Returns: {"answer": "...", "sources": [...], "num_sources": 4}
    """
```
- Takes a question string
- Returns structured data (perfect for APIs)
- Includes source attribution

```python
def ask_question(rag_chain, retriever, question):
    """
    Display-friendly version for terminal use.
    Prints formatted answer and sources.
    """
```
- Formats output nicely for human reading
- Shows page numbers and file names
- Displays source content previews

```python
def interactive_mode(rag_chain, retriever):
    """
    Loop that keeps asking for questions until you type 'quit'.
    """
```
- Continuous conversation interface
- Type 'quit', 'exit', or 'q' to stop

**Switching RAG Versions**:

The file has commented imports at the top:
```python
#from setup_rag import setup_rag
#from setup_rag_multiquery import setup_rag_multiquery
#from setup_rag_rerank import setup_rag_with_reranking
from setup_rag_hybrid import setup_rag_hybrid  # Currently active
```

To use a different version:
1. Comment out the current import (add `#`)
2. Uncomment the version you want (remove `#`)
3. Update the function call in `main()`

---

### STEP 6: FastAPI Web Endpoint

**File**: `api.py`

**What it does**: Creates a web API that other applications can call.

**Why?** So you can:
- Build a web interface for your chatbot
- Integrate with mobile apps
- Call from other programming languages
- Share with team members over network

**Run it**:
```bash
python api.py
```

Or:
```bash
uvicorn api:app --reload
```

**What happens**:
1. Server starts on `http://localhost:8000`
2. RAG system initializes once at startup
3. Waits for questions via HTTP requests

**Code Breakdown**:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
```
- Creates a FastAPI application
- FastAPI auto-generates API documentation

```python
class Query(BaseModel):
    question: str
```
- Defines the expected input format
- Pydantic validates that incoming requests have a "question" field

```python
@app.on_event("startup")
async def startup():
    initialize_rag()
```
- Runs once when server starts
- Loads the RAG system into memory
- Prevents reloading on every request (much faster)

```python
@app.post("/query")
async def query(q: Query):
    return query_rag(q.question)
```
- Defines an endpoint at `/query`
- Accepts POST requests with JSON: `{"question": "..."}`
- Returns JSON: `{"answer": "...", "sources": [...], "num_sources": 4}`

**How to use the API**:

1. **From command line** (using curl):
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?"}'
```

2. **From Python**:
```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={"question": "What is machine learning?"}
)
print(response.json())
```

3. **From browser** (interactive docs):
- Go to `http://localhost:8000/docs`
- See auto-generated API documentation
- Try requests directly from the browser

**API Response Format**:
```json
{
  "answer": "Machine learning is a subset of artificial intelligence...",
  "sources": [
    {
      "page": 5,
      "file": "Foundations_of_Machine_Learning.pdf",
      "content": "Machine learning (ML) is the study of..."
    },
    ...
  ],
  "num_sources": 4
}
```

---

### STEP 7: Enhanced API with PDF Upload

**File**: `api_enhanced.py`

**What it does**: Extends the basic API to allow uploading new PDFs through the web interface.

**Why?** This makes your RAG system dynamic:
- Upload PDFs without restarting the server
- Build a knowledge base incrementally
- Users can add documents through the web
- No need for command-line access

**Run it**:
```bash
python api_enhanced.py
```

Or:
```bash
uvicorn api_enhanced:app --reload
```

**What happens**:
1. Server starts on `http://localhost:8000`
2. RAG system initializes with existing database
3. Accepts both queries AND PDF uploads

**Key Features**:

#### 1. Query Endpoint (same as before)
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?"}'
```

#### 2. Upload Endpoint (NEW!)
Upload a PDF file to add it to the knowledge base:

**From command line**:
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@/path/to/your/document.pdf"
```

**From Python**:
```python
import requests

# Upload a PDF
with open("my_document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/upload",
        files={"file": f}
    )
print(response.json())
# Output: {
#   "filename": "my_document.pdf",
#   "status": "success",
#   "message": "PDF indexed and added to database. RAG system will reload on next query."
# }
```

**From browser**:
- Go to `http://localhost:8000/docs`
- Click on `POST /upload`
- Click "Try it out"
- Choose a PDF file
- Click "Execute"

**How Upload Works**:

```python
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # 1. Save the uploaded file
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. Index the PDF (adds to existing database)
    index_pdfs(str(file_path))

    # 3. Reset RAG system so it reloads with new documents
    ask_questions._rag_chain = None
    ask_questions._retriever = None

    return {"filename": file.filename, "status": "success"}
```

**What happens step-by-step**:

1. **File Upload**: PDF is saved to `./uploaded_pdfs/` folder
2. **Indexing**: `index_pdfs()` from `Indexer.py` processes the PDF:
   - Loads all pages
   - Splits into chunks (1000 chars, 200 overlap)
   - Creates embeddings
   - **Adds to existing ChromaDB** (doesn't replace!)
3. **RAG Reset**: Sets `_rag_chain` and `_retriever` to `None`
4. **Auto-Reload**: Next query will automatically reinitialize RAG with all documents

**Important Behavior**:

**Cumulative Knowledge Base**:
- Each upload **adds** to the database
- All previously uploaded PDFs remain searchable
- The knowledge base grows over time

Example workflow:
```python
# Upload first PDF
upload("machine_learning.pdf")   # Database has: ML.pdf

# Upload second PDF
upload("deep_learning.pdf")      # Database has: ML.pdf + DL.pdf

# Query searches BOTH documents
query("What is neural network?") # Searches across both PDFs
```

**Clearing the Database**:

If you want to start fresh:
```bash
# Stop the server first
# Then delete the database folder
rm -rf chroma_db/
rm -rf uploaded_pdfs/

# Restart the server
python api_enhanced.py
```

**Use Cases**:

1. **Document Management System**: Users upload company docs, ask questions
2. **Research Assistant**: Upload papers, query across all research
3. **Customer Support**: Upload product manuals, answer customer questions
4. **Personal Knowledge Base**: Upload books, notes, articles

**Comparison with Basic API**:

| Feature | `api.py` (Basic) | `api_enhanced.py` (Enhanced) |
|---------|------------------|------------------------------|
| Query documents | ✅ Yes | ✅ Yes |
| Upload PDFs | ❌ No | ✅ Yes |
| Dynamic updates | ❌ No (requires restart) | ✅ Yes (automatic) |
| Cumulative database | ❌ Manual only | ✅ Automatic |
| User-friendly | Medium | High |

**API Endpoints Summary**:

```
GET  /          - API information
POST /query     - Ask questions (same as basic API)
POST /upload    - Upload and index a PDF file
GET  /docs      - Interactive API documentation (Swagger UI)
```

**Example Complete Workflow**:

```python
import requests

API_URL = "http://localhost:8000"

# 1. Upload first PDF
with open("document1.pdf", "rb") as f:
    response = requests.post(f"{API_URL}/upload", files={"file": f})
    print(response.json())
# Output: {"filename": "document1.pdf", "status": "success", ...}

# 2. Upload second PDF
with open("document2.pdf", "rb") as f:
    response = requests.post(f"{API_URL}/upload", files={"file": f})
    print(response.json())
# Output: {"filename": "document2.pdf", "status": "success", ...}

# 3. Query across both documents
response = requests.post(
    f"{API_URL}/query",
    json={"question": "Summarize the main points from both documents"}
)
print(response.json()["answer"])
# Answer will include information from both PDFs

# 4. Check sources to see which documents were used
for source in response.json()["sources"]:
    print(f"From {source['file']}, page {source['page']}")
```

**Code Breakdown**:

The enhanced API reuses the `index_pdfs()` function from `Indexer.py`:

```python
from Indexer import index_pdfs

# This function handles:
# - Loading the PDF
# - Splitting into chunks
# - Creating embeddings
# - Adding to ChromaDB
```

**Why reuse instead of rewrite?**
- DRY principle (Don't Repeat Yourself)
- Consistent behavior with manual indexing
- Less code to maintain
- Proven and tested logic

**Automatic RAG Reinitialization**:

The system automatically detects when new documents are added:

```python
# In ask_questions.py
def query_rag(question: str) -> dict:
    global _rag_chain, _retriever

    # Initialize if not already done
    if _rag_chain is None:
        initialize_rag()  # Loads all documents from ChromaDB

    # Query with latest data
    answer = _rag_chain.invoke(question)
    ...
```

After upload, the next query triggers reinitialization, loading all documents including newly uploaded ones.

**Performance Notes**:

- **Upload time**: Depends on PDF size
  - Small (1-10 pages): 1-5 seconds
  - Medium (10-50 pages): 5-15 seconds
  - Large (50+ pages): 15+ seconds

- **First query after upload**: Slightly slower (RAG initialization)
- **Subsequent queries**: Normal speed

**Security Considerations** (for production):

The current implementation is for learning/development. For production, add:

1. **File validation**: Check file size, type
2. **Authentication**: Require API keys
3. **Rate limiting**: Prevent abuse
4. **Storage limits**: Cap database size
5. **Sanitization**: Clean filenames

Example additions:
```python
# Add file size check
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

if len(await file.read()) > MAX_FILE_SIZE:
    raise HTTPException(400, "File too large")
```

---

### STEP 8: Web Front-End Interface

**Files**: `index.html`, `style.css`, `app.js`

**What it does**: Provides a user-friendly web interface to interact with your RAG system without writing code.

**Why?** Makes your chatbot accessible to non-technical users:
- No command line needed
- Point-and-click interface
- Visual feedback
- Easy to use and share

**Project Structure**:
```
frontend/
├── index.html    # HTML structure
├── style.css     # Styling
└── app.js        # JavaScript logic
```

**How to Use**:

1. **Start the API server**:
```bash
python api_enhanced.py
```

2. **Open the front-end**:
```bash
open index.html
```
Or just double-click `index.html` in your file browser.

3. **Use the interface**:
   - Click "Choose File" to select a PDF
   - Click "Upload" to index it
   - Type a question in the text box
   - Click "Ask" to get an answer

**File Breakdown**:

#### `index.html` - Structure

```html
<!DOCTYPE html>
<html>
<head>
    <title>PDF RAG Chatbot</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <h1>PDF RAG Chatbot</h1>

    <!-- Upload Section -->
    <div class="section">
        <h2>Upload PDF</h2>
        <input type="file" id="fileInput" accept=".pdf">
        <button onclick="uploadPDF()">Upload</button>
        <div id="uploadStatus"></div>
    </div>

    <!-- Query Section -->
    <div class="section">
        <h2>Ask Question</h2>
        <input type="text" id="questionInput" placeholder="Enter your question...">
        <button onclick="askQuestion()">Ask</button>
        <div id="queryResult"></div>
    </div>

    <script src="app.js"></script>
</body>
</html>
```

**What each part does**:
- `<link rel="stylesheet" href="style.css">`: Links to external CSS file
- `<input type="file" accept=".pdf">`: File picker that only accepts PDFs
- `onclick="uploadPDF()"`: Calls JavaScript function when button is clicked
- `<div id="uploadStatus">`: Empty div where upload status will appear
- `<script src="app.js">`: Links to external JavaScript file

---

#### `style.css` - Styling

```css
body {
    font-family: Arial, sans-serif;
    max-width: 800px;
    margin: 50px auto;
    padding: 20px;
}

.section {
    margin: 30px 0;
    padding: 20px;
    border: 1px solid #ddd;
    border-radius: 5px;
}

input[type="file"],
input[type="text"] {
    padding: 8px;
    margin: 10px 0;
    width: 70%;
}

button {
    padding: 10px 20px;
    background: #4CAF50;
    color: white;
    border: none;
    cursor: pointer;
    border-radius: 3px;
}

.result {
    margin-top: 20px;
    padding: 15px;
    background: #f4f4f4;
    border-radius: 5px;
}

.source {
    margin: 10px 0;
    padding: 10px;
    background: white;
    border-left: 3px solid #4CAF50;
}
```

**What this does**:
- Centers content with `max-width: 800px; margin: auto`
- Styles sections with borders and padding
- Makes buttons green (`#4CAF50`) with rounded corners
- Styles results and sources for readability

---

#### `app.js` - JavaScript Logic

```javascript
const API_URL = 'http://localhost:8000';

async function uploadPDF() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];

    if (!file) {
        alert('Please select a PDF file');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    document.getElementById('uploadStatus').innerHTML = 'Uploading...';

    try {
        const response = await fetch(`${API_URL}/upload`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        document.getElementById('uploadStatus').innerHTML =
            `<div class="result">${data.message}</div>`;
    } catch (error) {
        document.getElementById('uploadStatus').innerHTML =
            `<div class="result">Error: ${error.message}</div>`;
    }
}

async function askQuestion() {
    const question = document.getElementById('questionInput').value;

    if (!question) {
        alert('Please enter a question');
        return;
    }

    document.getElementById('queryResult').innerHTML = 'Thinking...';

    try {
        const response = await fetch(`${API_URL}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: question })
        });

        const data = await response.json();

        let html = `<div class="result"><strong>Answer:</strong><p>${data.answer}</p>`;

        if (data.sources) {
            html += '<strong>Sources:</strong>';
            data.sources.forEach(source => {
                html += `<div class="source">
                    ${source.file} (Page ${source.page})<br>
                    <small>${source.content}</small>
                </div>`;
            });
        }
        html += '</div>';

        document.getElementById('queryResult').innerHTML = html;
    } catch (error) {
        document.getElementById('queryResult').innerHTML =
            `<div class="result">Error: ${error.message}</div>`;
    }
}
```

**How this works**:

1. **Upload Function** (`uploadPDF`):
   - Gets the selected file from the input
   - Creates `FormData` object (required for file uploads)
   - Sends POST request to `/upload` endpoint
   - Displays result or error message

2. **Query Function** (`askQuestion`):
   - Gets the question text from input
   - Sends POST request to `/query` endpoint with JSON
   - Displays answer and sources
   - Handles errors gracefully

**Key JavaScript Concepts**:

- **`async/await`**: Modern way to handle asynchronous operations
- **`fetch()`**: Built-in function to make HTTP requests
- **`FormData`**: Special object for uploading files
- **`JSON.stringify()`**: Converts JavaScript object to JSON string
- **`document.getElementById()`**: Gets HTML element by its ID
- **`.innerHTML`**: Sets the HTML content inside an element

---

### CORS: Why It's Needed

When you first try to use the front-end, you'll get this error:

```
Access to fetch at 'http://localhost:8000/query' from origin 'null'
has been blocked by CORS policy
```

**What is CORS?**

CORS stands for **Cross-Origin Resource Sharing**. It's a browser security feature.

**The Problem**:
- Your HTML file is served from `file://` (local file system)
- Your API runs on `http://localhost:8000` (web server)
- These are different "origins"
- Browsers block requests between different origins by default (for security)

**The Solution**:

Add CORS middleware to your FastAPI server:

```python
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)
```

**What this does**:
- `allow_origins=["*"]`: Accept requests from any origin
  - For production, specify exact origins: `["http://localhost:3000"]`
- `allow_methods=["*"]`: Allow all HTTP methods (GET, POST, PUT, etc.)
- `allow_headers=["*"]`: Allow all request headers

**Security Note**:

Using `["*"]` (allow all) is fine for development and local use. For production:

```python
# Production example - more restrictive
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://yourdomain.com",
        "https://www.yourdomain.com"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

**Why browsers enforce CORS**:

Imagine a malicious website trying to:
1. Make requests to your bank's website
2. Access your logged-in session
3. Steal your data or make transactions

CORS prevents this by:
- Blocking cross-origin requests by default
- Only allowing them when the server explicitly permits it
- Protecting users from malicious websites

**In our case**:
- We control both the front-end and API
- We want them to communicate
- So we explicitly enable CORS

---

### Complete Workflow Example

Here's how everything works together:

1. **User opens `index.html` in browser**
   - Browser loads HTML, CSS, and JavaScript
   - Page displays upload and query sections

2. **User uploads a PDF**:
   ```
   Browser (index.html)
   ↓ User selects file
   ↓ Clicks "Upload"
   ↓ JavaScript (app.js) runs uploadPDF()
   ↓ Creates FormData with file
   ↓ fetch() sends POST to http://localhost:8000/upload
   ↓
   API Server (api_enhanced.py)
   ↓ Receives file
   ↓ Saves to ./uploaded_pdfs/
   ↓ Calls index_pdfs() from Indexer.py
   ↓ Indexes PDF to ChromaDB
   ↓ Returns success response
   ↓
   Browser
   ↓ Displays "PDF indexed successfully"
   ```

3. **User asks a question**:
   ```
   Browser (index.html)
   ↓ User types question
   ↓ Clicks "Ask"
   ↓ JavaScript (app.js) runs askQuestion()
   ↓ fetch() sends POST to http://localhost:8000/query
   ↓
   API Server (api_enhanced.py)
   ↓ Calls query_rag() from ask_questions.py
   ↓ Initializes RAG if needed
   ↓ Searches ChromaDB
   ↓ Generates answer with LLM
   ↓ Returns answer + sources
   ↓
   Browser
   ↓ JavaScript builds HTML from response
   ↓ Displays answer and sources
   ```

**Error Handling**:

Both JavaScript functions include `try/catch` blocks:

```javascript
try {
    // Try to do something
    const response = await fetch(...);
} catch (error) {
    // If it fails, show error to user
    document.getElementById('result').innerHTML =
        `Error: ${error.message}`;
}
```

This ensures users see helpful error messages instead of a broken interface.

---

### Testing the Front-End

1. **Start the API**:
```bash
python api_enhanced.py
```

You should see:
```
RAG system initialized!
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

2. **Open the front-end**:
```bash
open index.html
```

3. **Test upload**:
   - Click "Choose File"
   - Select a PDF
   - Click "Upload"
   - Wait for "PDF indexed successfully" message

4. **Test query**:
   - Type: "What is this document about?"
   - Click "Ask"
   - Wait for answer to appear
   - Check that sources are displayed

5. **Test error handling**:
   - Stop the API server (CTRL+C)
   - Try to ask a question
   - You should see an error message (not a broken page)

---

### Troubleshooting Front-End Issues

#### Problem: "Failed to fetch" error

**Cause**: API server not running

**Solution**:
```bash
# Start the API server
python api_enhanced.py
```

---

#### Problem: "CORS policy" error

**Cause**: CORS middleware not added to API

**Solution**: Ensure `api_enhanced.py` has CORS middleware:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Restart the server after adding this.

---

#### Problem: Upload succeeds but query returns no results

**Cause**: RAG system not reinitialized after upload

**Solution**: This should happen automatically. Check:
1. Upload shows success message
2. First query after upload might be slow (RAG reinitializing)
3. Check API server logs for errors

---

#### Problem: Sources not displaying

**Cause**: API response format changed

**Solution**: Check API response in browser console (F12):
```javascript
// Should look like:
{
  "answer": "...",
  "sources": [
    {"file": "...", "page": 0, "content": "..."}
  ],
  "num_sources": 4
}
```

---

### Enhancing the Front-End (Ideas for Later)

Once comfortable with the basics, you can add:

1. **Loading spinners**: Visual feedback during upload/query
2. **Upload progress bar**: Show upload percentage
3. **Question history**: Keep track of previous questions
4. **Copy answer button**: Easy to copy answers
5. **Dark mode**: Toggle between light/dark themes
6. **Markdown rendering**: Format answers with headings, lists, etc.
7. **File list**: Show all uploaded PDFs
8. **Delete files**: Remove PDFs from database
9. **Settings panel**: Adjust temperature, model, k value
10. **Export conversation**: Save Q&A as PDF or text file

Example loading spinner:
```javascript
// Add to app.js
function showLoading() {
    return '<div class="loading">⏳ Loading...</div>';
}

// Use it
document.getElementById('queryResult').innerHTML = showLoading();
```

---

## Design Decisions Explained

This section explains **why** the code is written the way it is. Understanding these decisions will help you make informed choices in your own projects.

---

### Why LCEL Instead of RetrievalQA?

You might notice the code uses **LCEL (LangChain Expression Language)** syntax:

```python
# LCEL approach (what we use)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

Instead of the older **RetrievalQA** class:

```python
# Legacy approach (not used)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
```

**Why LCEL is better for learning and production:**

#### 1. **Transparency**
LCEL shows exactly what's happening at each step:
- `retriever` → searches for documents
- `format_docs` → formats them into text
- `prompt` → adds instructions
- `llm` → generates answer
- `StrOutputParser()` → extracts the text

With `RetrievalQA`, these steps are hidden inside `chain_type="stuff"`.

#### 2. **Easy to Customize**
Want to add a step? Just insert it:

```python
# Add filtering between retrieval and formatting
rag_chain = (
    {"context": retriever | filter_function | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

With `RetrievalQA`, you'd need to subclass or hack around the framework.

#### 3. **Modern Standard**
- LCEL is the **current recommended approach** (since LangChain 0.1.0, January 2024)
- `RetrievalQA` still works but is considered **legacy**
- New LangChain features are built for LCEL first

#### 4. **Better Streaming**
```python
# Streaming is built-in with LCEL
for chunk in rag_chain.stream("What is machine learning?"):
    print(chunk, end="", flush=True)
```

This gives you real-time output as the LLM generates text (like ChatGPT's typing effect).

#### 5. **Composability**
LCEL chains are **Runnables** - you can combine them like building blocks:

```python
chain_a = retriever | format_docs
chain_b = prompt | llm
final_chain = chain_a | chain_b  # Easy composition
```

#### 6. **Automatic Parallelization**
LCEL runs independent steps in parallel:

```python
{
    "context": retriever | format_docs,  # Runs in parallel
    "question": RunnablePassthrough()     # with this
}
```

This improves performance automatically.

**When to use RetrievalQA:**
- Quick prototypes where you don't care about customization
- Maintaining legacy code that already uses it
- You prefer less verbose code

**Bottom line:** LCEL is more transparent, flexible, and future-proof. Perfect for learning and production code.

---

### Why These Specific Models?

#### Embedding Model: `qwen3-embedding:0.6b`

**Why this model?**
1. **Small size**: 600 million parameters ≈ 600MB on disk
   - Runs on most computers (even laptops without GPU)
   - Downloads quickly
2. **Fast inference**: Can embed thousands of chunks in seconds
3. **Good quality**: Competitive with larger models for general knowledge tasks
4. **Free and local**: No API costs, no internet required

**Alternatives you could try:**
- `nomic-embed-text`: More popular, slightly better accuracy, similar size
- `mxbai-embed-large`: Larger (1.5GB), better quality, slower
- OpenAI embeddings (via API): Higher quality, costs money, requires internet

**How to change:**
```python
# In Indexer.py and all setup_rag*.py files
embeddings = OllamaEmbeddings(model="nomic-embed-text")
```

**Important:** You must re-run `Indexer.py` if you change the embedding model!

#### Language Model: `llama3.2:1b`

**Why this model?**
1. **Lightweight**: 1 billion parameters ≈ 1.3GB
   - Runs smoothly on CPU
   - Fast response times (1-2 seconds)
2. **Capable enough**: Good at following instructions and answering questions
3. **Latest generation**: Released by Meta in 2024
4. **Free and local**: No API costs

**Alternatives you could try:**
- `llama3.2:3b`: Better quality, still lightweight (3GB)
- `llama3:8b`: Much better quality, needs more RAM (8GB)
- `qwen2.5:7b`: Excellent quality, good for technical content
- `gemma2:9b`: Google's model, very capable
- Claude API / GPT-4: Highest quality, costs money

**How to change:**
```python
# In all setup_rag*.py files
llm = Ollama(model="llama3:8b", temperature=0.7)
```

**Trade-offs:**
- **Smaller models** (1B-3B): Fast, lightweight, good for simple Q&A
- **Medium models** (7B-9B): Better reasoning, more accurate, slower
- **Large models** (70B+): Best quality, need powerful hardware or cloud APIs

---

### Why ChromaDB?

**Chosen because:**

1. **Embedded database**: No separate server needed
   - Just a Python package: `pip install chromadb`
   - Data stored in local folder (`./chroma_db/`)
   - Perfect for development and small-to-medium projects

2. **Persistent storage**: Data saved to disk
   - Build index once, reuse many times
   - Survives restarts

3. **Simple API**: Easy to use
   ```python
   vectorstore = Chroma.from_documents(documents, embeddings)
   results = vectorstore.similarity_search("query", k=4)
   ```

4. **Good performance**: HNSW indexing
   - Fast approximate nearest neighbor search
   - Handles tens of thousands of documents easily

5. **Metadata filtering**: Search within subsets
   ```python
   results = vectorstore.similarity_search(
       "query",
       filter={"page": 5}
   )
   ```

**Alternatives you could consider:**

| Database | Best For | Trade-offs |
|----------|----------|------------|
| **FAISS** | Maximum speed, millions of vectors | No persistence (in-memory), no metadata filtering |
| **Pinecone** | Production, cloud-hosted, scale | Costs money, requires internet |
| **Weaviate** | Complex queries, production | Needs separate server |
| **Qdrant** | Production, on-premise | Needs separate server |
| **PostgreSQL + pgvector** | Already use Postgres, familiar SQL | Slower than specialized vector DBs |

**Bottom line:** ChromaDB is perfect for learning and for projects with up to ~100K chunks. For millions of vectors or production scale, consider FAISS or cloud options.

---

### Why This Progression of RAG Techniques?

The project shows **4 different RAG approaches** in increasing complexity:

#### Step 1: Basic RAG
**When to use:**
- Simple Q&A on focused documents
- When speed is critical
- Documents have clear, direct information

**Limitations:**
- Single query phrasing might miss relevant info
- No quality control on retrieved chunks
- Pure semantic search misses exact term matches

#### Step 2: MultiQuery RAG
**Adds:**
- Query variation generation
- More comprehensive retrieval

**When to use:**
- Questions that can be phrased many ways
- Documents use varied terminology
- Want better recall (finding all relevant info)

**Trade-off:**
- Slower (3x the searches + LLM call for query generation)
- But significantly better at finding all relevant content

#### Step 3: RAG with Reranking
**Adds:**
- Quality control via reranking model
- Better precision (top results are more relevant)

**When to use:**
- Need high accuracy
- Initial retrieval brings in some irrelevant results
- Have slightly more compute budget

**Trade-off:**
- Additional latency (~100-200ms)
- Needs FlashRank model downloaded
- But much better answer quality

#### Step 4: Hybrid RAG (Recommended for Production)
**Adds:**
- BM25 keyword search
- Best of both worlds (semantic + keyword)

**When to use:**
- Production systems
- Documents with technical terms, acronyms, names
- Need both conceptual and exact matching

**Trade-off:**
- Most complex code
- Slightly slower than basic RAG
- But highest quality and most robust

**Which should you use?**
- **Learning:** Start with Basic (Step 1), progress through all
- **Prototype:** MultiQuery (Step 2) is often sufficient
- **Production:** Hybrid (Step 4) gives best results

---

### Why These Chunking Parameters?

In `Indexer.py`:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
```

#### `chunk_size=1000`

**Why 1000 characters?**
- **Not too small**: Enough context for meaningful retrieval
  - ~150-200 words
  - Typically 2-4 paragraphs
- **Not too large**: Fits in LLM context with room for multiple chunks
  - 4 chunks × 1000 chars = 4000 chars ≈ 3000 tokens
  - Leaves room for question + prompt + answer in context window
- **Balance**: Good trade-off between precision and context

**Alternatives:**
- **500 chars**: More precise, but fragments concepts
- **2000 chars**: More context, but less precise retrieval
- **Semantic chunking**: Split by topics (more advanced)

#### `chunk_overlap=200`

**Why 200 character overlap?**
- **Prevents loss at boundaries**: Important info spanning chunk edges isn't lost
- **20% overlap**: Good balance
  - Too little (50): Risk missing cross-boundary info
  - Too much (500): Redundancy, larger database

**Example:**
```
Chunk 1: "...machine learning is a subset of AI. It uses algorithms to..."
                                                    ↓ 200 char overlap
Chunk 2: "...algorithms to learn from data. Neural networks are..."
```

The overlap ensures "algorithms" context appears in both chunks.

#### `length_function=len`

Uses Python's `len()` to count characters (not tokens).
- **Simple**: Easy to reason about
- **Fast**: No tokenization needed during chunking
- **Good enough**: Character count approximates token count reasonably well

**Alternative:**
```python
# Token-based chunking (more accurate for LLM limits)
from tiktoken import get_encoding
encoding = get_encoding("cl100k_base")
length_function = lambda text: len(encoding.encode(text))
```

---

### Why `temperature=0.7`?

In all LLM configurations:

```python
llm = Ollama(model="llama3.2:1b", temperature=0.7)
```

**What is temperature?**
- Controls randomness in text generation
- Range: 0.0 (deterministic) to 2.0 (very random)

**Temperature scale:**
- **0.0**: Always picks most likely word
  - Deterministic, repeatable
  - Can be robotic or repetitive
  - Good for: Classification, extraction, structured output

- **0.7** (our choice): Balanced
  - Mostly coherent, slightly creative
  - Natural-sounding answers
  - Good for: Q&A, explanations, conversation

- **1.0+**: More creative/random
  - More diverse outputs
  - Can hallucinate more
  - Good for: Creative writing, brainstorming

**Why 0.7 for RAG?**
- Answers sound natural, not robotic
- Still grounded (not too creative)
- Slight variation makes it less repetitive
- Industry standard for Q&A tasks

**Experiment:**
```python
# More deterministic
llm = Ollama(model="llama3.2:1b", temperature=0.2)

# More creative
llm = Ollama(model="llama3.2:1b", temperature=1.0)
```

---

### Why `k=4` Retrieved Chunks?

In retrieval:

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
```

**Why 4 chunks?**

1. **Context window limits**:
   - Small models have limited context (2K-8K tokens)
   - 4 chunks × 1000 chars ≈ 3000 tokens
   - Leaves room for: question + prompt + answer generation

2. **Diminishing returns**:
   - Top 4 chunks usually contain the answer
   - Additional chunks often add noise
   - More chunks = longer processing time

3. **Quality over quantity**:
   - Better to have 4 highly relevant chunks
   - Than 10 chunks with 6 irrelevant ones

**When to adjust:**

**Increase to 6-8 if:**
- Information is scattered across document
- Complex questions need multiple perspectives
- Using reranking (can filter down after)

**Decrease to 2-3 if:**
- Very focused documents
- Simple questions
- Small context window LLM

**Hybrid RAG uses different k:**
```python
# Retrieve more initially because reranking will filter
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

# Rerank down to 4
compressor = FlashrankRerank(top_n=4)
```

---

### Why These Specific Libraries?

#### LangChain
**Why:**
- Industry standard for RAG applications
- Huge ecosystem of integrations
- Active development and community
- Good documentation

**Alternatives:**
- **LlamaIndex**: More focused on RAG, sometimes easier
- **Haystack**: Similar to LangChain, good alternative
- **Build from scratch**: More control, more code

#### FastAPI (for API)
**Why:**
- Modern, fast Python web framework
- Automatic API documentation (Swagger UI)
- Type hints and validation built-in
- Async support for better performance

**Alternatives:**
- **Flask**: Simpler, older, more verbose
- **Django**: Overkill for simple API
- **Gradio/Streamlit**: Better for UI, not APIs

#### Ollama
**Why:**
- Easy local model management
- One-command model downloads
- Works across platforms
- Handles model serving automatically

**Alternatives:**
- **llama.cpp**: More control, harder to use
- **HuggingFace Transformers**: Direct model loading, more complex
- **Cloud APIs**: Easier, costs money

---

## Key Takeaways

These design decisions prioritize:

1. **Learning**: Code is transparent and understandable
2. **Practicality**: Runs on consumer hardware
3. **Modern practices**: Uses current best practices (LCEL, etc.)
4. **Flexibility**: Easy to modify and experiment
5. **Cost**: Everything runs locally for free

As you gain experience, you can make different trade-offs based on your specific needs:
- **More accuracy?** Use larger models or cloud APIs
- **More scale?** Switch to production vector databases
- **Less complexity?** Use simpler RAG without hybrid search

The goal is to understand the fundamentals so you can make informed decisions.

---

## Detailed Module Explanations

### What is an Embedding?

**Simple explanation**: A way to represent text as numbers so computers can understand meaning.

**Example**:
- "dog" → [0.8, 0.3, -0.2, 0.5, ...]
- "puppy" → [0.7, 0.4, -0.1, 0.6, ...] (similar numbers because similar meaning)
- "car" → [-0.3, 0.9, 0.7, -0.4, ...] (very different numbers)

**Technical explanation**:
- Embedding models are neural networks trained on massive text datasets
- They map words/sentences to high-dimensional vectors (typically 384-1536 dimensions)
- Mathematically similar vectors represent semantically similar content
- We use cosine similarity to compare vectors

**Why we use `qwen3-embedding:0.6b`**:
- Small (600MB) - runs on most computers
- Fast - can embed thousands of chunks quickly
- Good quality for general knowledge tasks

---

### What is a Vector Database?

**Simple explanation**: A specialized database for storing and searching embeddings.

**Regular database**:
- Stores exact data
- Searches for exact matches
- Example: Find all users named "John"

**Vector database**:
- Stores embeddings (vectors)
- Searches for similar vectors
- Example: Find text similar to "machine learning basics"

**How ChromaDB works**:

1. **Storage**:
   - Saves vectors to disk (`chroma_db/` folder)
   - Includes metadata (page number, source file)
   - Persists between runs

2. **Indexing**:
   - Builds an index for fast similarity search
   - Uses HNSW (Hierarchical Navigable Small World) algorithm
   - Trade-off: Approximate but very fast

3. **Querying**:
   - Takes your question embedding
   - Finds nearest neighbors in vector space
   - Returns top K most similar chunks

**Why ChromaDB**:
- Lightweight (no separate server needed)
- Easy to set up (just a Python package)
- Good enough for thousands of documents
- Free and open source

---

### What is an LLM (Large Language Model)?

**Simple explanation**: An AI model trained to understand and generate human language.

**How it works**:
1. Trained on massive amounts of text from the internet
2. Learns patterns, grammar, facts, and reasoning
3. Can predict what text should come next
4. Can follow instructions and answer questions

**We use `llama3.2:1b`**:
- Small (1 billion parameters)
- Runs locally on your computer
- No internet needed
- Free to use

**Temperature parameter**:
- `temperature=0.7`: Controls randomness
- `0.0`: Very deterministic, always picks most likely word
- `1.0`: More creative, more random choices
- `0.7`: Balanced - coherent but not robotic

---

### What is BM25?

**Simple explanation**: A classic keyword-based search algorithm (like Google in the 1990s).

**How it scores documents**:

```
Score = TF (Term Frequency) × IDF (Inverse Document Frequency)
```

**Example**:
- Query: "machine learning algorithms"
- Document 1: Mentions "machine" 5 times, "learning" 3 times, "algorithms" 2 times
- Document 2: Mentions "machine" 1 time, "learning" 1 time, "algorithms" 10 times

**TF (Term Frequency)**:
- How often does the term appear in this document?
- More appearances = higher score
- But with diminishing returns (10 mentions not 10x better than 1)

**IDF (Inverse Document Frequency)**:
- How rare is this term across all documents?
- Rare terms are more important
- Common words like "the" have low IDF

**Why combine with vector search?**:
- BM25 is great for rare/specific terms
- Vector search is great for concepts/meaning
- Together, they complement each other

---

### What is Reranking?

**Simple explanation**: A second pass that re-sorts search results for better accuracy.

**Why needed?**

Initial retrieval (BM25 or vector search):
- Fast but approximate
- May rank items incorrectly
- Uses simple similarity metrics

Reranking:
- Slower but more accurate
- Uses a specialized model
- Considers query-document interaction

**FlashRank Model**:
- A lightweight neural reranker
- Trained specifically for relevance scoring
- Reads both query and document
- Outputs a relevance score

**Process**:
1. Retrieve 10 potentially relevant chunks (cast wide net)
2. Reranker scores each chunk against your question
3. Sort by score
4. Keep only top 4 (highest quality)

**Trade-off**: Adds ~100-200ms latency but significantly improves accuracy.

---

### What is LangChain?

**Simple explanation**: A framework that makes building AI applications easier.

**Without LangChain**:
```python
# You'd have to manually:
embeddings = create_embeddings(question)
results = vectordb.search(embeddings)
context = format_results(results)
prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
answer = llm.generate(prompt)
```

**With LangChain**:
```python
# Clean, composable chains:
chain = retriever | format_docs | prompt | llm | parser
answer = chain.invoke(question)
```

**Key LangChain concepts**:

1. **Chains**: Connect components in a pipeline
2. **Retrievers**: Standardized interface for search
3. **Prompts**: Templates for LLM instructions
4. **Output Parsers**: Process LLM responses
5. **LCEL**: LangChain Expression Language - the `|` syntax

**Benefits**:
- Less boilerplate code
- Easier to modify and experiment
- Built-in observability and debugging
- Large ecosystem of integrations

---

### Understanding the Prompt Template

```python
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer based on the context, just say that you don't know,
don't try to make up an answer.

Context:
{context}

Question: {question}

Answer: """
```

**Why this prompt?**

1. **"Use the following pieces of context"**:
   - Tells LLM to focus on provided information
   - Prevents hallucination (making things up)

2. **"If you don't know... don't make up an answer"**:
   - Critical instruction
   - LLMs tend to generate plausible-sounding but wrong answers
   - This reduces false information

3. **Variables**:
   - `{context}`: Replaced with retrieved document chunks
   - `{question}`: Replaced with user's question

4. **Format matters**:
   - Clear structure helps LLM understand its task
   - Consistent format produces consistent results

**Prompt engineering** is the art of writing instructions that get good LLM outputs.

---

## Testing & Troubleshooting

### Test the Vector Database

**File**: `test_vectordb.py`

**Run it**:
```bash
python test_vectordb.py
```

**What it tests**:

1. **Database Statistics**:
   - How many chunks are stored
   - Collection name and metadata

2. **Sample Entries**:
   - Shows first 3 chunks
   - Displays metadata (source, page)

3. **Similarity Search**:
   - Tests various queries
   - Shows top 3 results per query

4. **Search with Scores**:
   - Includes similarity scores
   - Lower score = more similar (ChromaDB uses distance metrics)

5. **Metadata Filtering**:
   - Search within specific pages
   - Search within specific files
   - Useful for narrowing results

**Use cases**:
- Verify indexing worked correctly
- Understand what's in your database
- Debug why certain queries don't work
- Explore available metadata for filtering

---

### Common Issues

#### 1. "No module named 'chromadb'"

**Problem**: Python package not installed.

**Solution**:
```bash
pip install -r requirements.txt
```

---

#### 2. "Cannot connect to Ollama"

**Problem**: Ollama service not running.

**Solution**:
- Mac: Make sure Ollama app is running
- Linux: Run `ollama serve`
- Check: `ollama list` should show your models

---

#### 3. "Model not found: qwen3-embedding:0.6b"

**Problem**: Embedding model not downloaded.

**Solution**:
```bash
ollama pull qwen3-embedding:0.6b
ollama pull llama3.2:1b
```

---

#### 4. "chromadb folder not found"

**Problem**: Need to run indexing first.

**Solution**:
```bash
python Indexer.py
```

---

#### 5. Slow response times

**Possible causes**:
1. Large PDF (many chunks to search)
2. Slow computer
3. Using large language models

**Solutions**:
- Use smaller models (`llama3.2:1b` instead of `llama3:8b`)
- Reduce `k` value in retriever (fewer chunks)
- Add more overlap to chunks (better results, but more chunks)

---

#### 6. Irrelevant answers

**Possible causes**:
1. Question not matching document content
2. Chunking broke up relevant information
3. Need better retrieval

**Solutions**:
- Try MultiQuery or Hybrid RAG (more advanced versions)
- Adjust chunk_size in Indexer.py
- Check if information actually exists in PDF

---

#### 7. "Address already in use" when running API

**Problem**: Port 8000 already used by another application.

**Solution**:
```bash
# Use a different port
uvicorn api:app --port 8001
```

Or kill the existing process:
```bash
# Mac/Linux
lsof -ti:8000 | xargs kill -9

# Windows
netstat -ano | findstr :8000
taskkill /PID <pid> /F
```

---

## Common Terms Glossary

**Chunk**: A small piece of text (typically 500-2000 characters) from a larger document.

**Embedding**: A numerical representation (vector) of text that captures its meaning.

**Vector**: A list of numbers, like [0.2, -0.5, 0.8, ...]. Used to represent text in multi-dimensional space.

**Cosine Similarity**: A measure of how similar two vectors are. Ranges from -1 (opposite) to 1 (identical).

**Retriever**: A component that searches and returns relevant documents/chunks.

**LLM (Large Language Model)**: An AI model trained to understand and generate text.

**Prompt**: Instructions given to an LLM about what to do.

**Temperature**: Parameter controlling randomness in LLM output. Lower = more predictable, higher = more creative.

**RAG (Retrieval-Augmented Generation)**: Technique combining search (retrieval) with AI generation.

**Semantic Search**: Search based on meaning, not just keywords. Uses embeddings.

**Keyword Search**: Traditional search looking for exact word matches. Uses algorithms like BM25.

**Hybrid Search**: Combines semantic and keyword search for best results.

**Reranking**: Re-sorting search results using a more sophisticated model.

**API (Application Programming Interface)**: A way for different programs to communicate. Lets you call functions over the web.

**FastAPI**: A Python framework for building web APIs quickly.

**JSON**: A data format for exchanging information between programs. Looks like: `{"key": "value"}`.

**Endpoint**: A specific URL path in an API, like `/query`.

**POST Request**: An HTTP request that sends data to a server.

**Metadata**: Additional information about data. For chunks: page number, source file, etc.

**Overlap**: When chunking, the number of characters shared between consecutive chunks. Prevents losing context at boundaries.

**Parameters**: Numbers in a neural network. More parameters = more capable, but slower and bigger.

**Local Model**: AI model running on your own computer, not in the cloud.

**Inference**: Running a trained model to get predictions/outputs.

**Context Window**: How much text an LLM can process at once. Measured in tokens (roughly 0.75 words per token).

**Hallucination**: When an LLM makes up false information that sounds plausible.

**Zero-shot**: Using an AI model without specific training on the task.

**Few-shot**: Giving a model examples before asking it to perform a task.

---

## Next Steps

### Experiment and Learn

1. **Try different questions**: Test the limits of your RAG system

2. **Use your own PDFs**:
   - Replace the PDF file
   - Update `Indexer.py` line 65 to your filename
   - Run `python Indexer.py` again

3. **Adjust parameters**:
   - Change chunk_size in `Indexer.py` (try 500, 1500)
   - Change `k` value (number of chunks retrieved)
   - Try different LLMs: `ollama pull llama3:8b`

4. **Compare RAG versions**:
   - Run the same question through all 4 versions
   - See which gives better answers
   - Understand the trade-offs

5. **Build a web interface**:
   - Use the FastAPI endpoint
   - Create a simple HTML page with JavaScript
   - Or use a framework like Streamlit

---

## Advanced Improvements (Ideas for Later)

Once you're comfortable with the basics:

1. **Query Expansion**: Add synonyms and related terms to queries
2. **Self-Querying**: Let LLM decide what metadata filters to use
3. **Parent Document Retrieval**: Retrieve small chunks but provide larger context to LLM
4. **Hypothetical Document Embeddings (HyDE)**: Generate an ideal answer first, then search for similar chunks
5. **Agentic RAG**: LLM decides when to retrieve, what to retrieve, and how many times
6. **Multi-vector Retrieval**: Store multiple embeddings per chunk (summary, questions it answers, etc.)
7. **Query Routing**: Route different types of questions to different retrievers or LLMs

---

## Resources for Further Learning

**LangChain Documentation**: https://python.langchain.com/
**Ollama Models**: https://ollama.ai/library
**ChromaDB Docs**: https://docs.trychroma.com/
**FastAPI Tutorial**: https://fastapi.tiangolo.com/tutorial/

**RAG Concepts**:
- Papers with Code: https://paperswithcode.com/task/question-answering
- Awesome RAG: https://github.com/awesome-rag/awesome-rag

---

## Conclusion

You've just built a production-quality RAG system from scratch! You've learned:

1. How to convert documents into searchable embeddings
2. Four different retrieval strategies and when to use them
3. How to combine multiple AI techniques for better results
4. How to expose AI functionality through web APIs

This foundation will help you build chatbots, document search systems, knowledge assistants, and more.

The key is to start simple (Basic RAG) and progressively add complexity (Hybrid + Reranking) as needed.

Happy building! 🚀
