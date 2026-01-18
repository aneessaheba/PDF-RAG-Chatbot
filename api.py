from fastapi import FastAPI
from pydantic import BaseModel
from ask_questions import query_rag, initialize_rag

app = FastAPI()

class Query(BaseModel):
    question: str

@app.on_event("startup")
async def startup():
    initialize_rag()

@app.post("/query")
async def query(q: Query):
    return query_rag(q.question)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
