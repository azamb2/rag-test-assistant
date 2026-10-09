from fastapi import FastAPI
from pydantic import BaseModel
from src.rag import ask_rag


app = FastAPI(
    title="RAG Test Intelligence API"
)


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "message": "RAG Test Intelligence API is running"
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):
	
    answer = ask_rag(request.question)
    return {
        "question": request.question,
        "answer": answer
    }
