from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import ask_questions
from Indexer import index_pdfs
import shutil
from pathlib import Path

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create upload directory
UPLOAD_DIR = Path("./uploaded_pdfs")
UPLOAD_DIR.mkdir(exist_ok=True)


class Query(BaseModel):
    question: str


@app.on_event("startup")
async def startup():
    try:
        ask_questions.initialize_rag()
    except:
        pass  # Will initialize after first PDF upload


@app.post("/query")
async def query(q: Query):
    """Ask questions about indexed documents."""
    return ask_questions.query_rag(q.question)


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and index a new PDF file (appends to existing database)."""

    # Save file
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Index the PDF (adds to existing database)
    index_pdfs(str(file_path))

    # Reset RAG system so it reloads with new documents
    ask_questions._rag_chain = None
    ask_questions._retriever = None

    return {
        "filename": file.filename,
        "status": "success",
        "message": "PDF indexed and added to database. RAG system will reload on next query."
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
