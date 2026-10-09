from fastapi.testclient import TestClient
from src.api import app


client = TestClient(app)


def test_home():

    response = client.get("/")

    assert response.status_code == 200

    assert response.json() == {
        "message": "RAG Test Intelligence API is running"
    }



def test_ask_question(monkeypatch):

    # Create a fake RAG function
    def fake_ask_rag(question):
        return "The account lockout period must last 30 minutes. Requirement ID: AUTH-004"


    # Replace the real ask_rag with our fake one
    monkeypatch.setattr(
        "src.api.ask_rag",
        fake_ask_rag
    )


    # Send a fake POST request
    response = client.post(
        "/ask",
        json={
            "question": "How long does an account stay locked?"
        }
    )


    # Check HTTP status
    assert response.status_code == 200


    # Check returned JSON
    assert response.json() == {
        "question": "How long does an account stay locked?",
        "answer": "The account lockout period must last 30 minutes. Requirement ID: AUTH-004"
    }


def test_ask_question_without_question():

    response = client.post(
        "/ask",
        json={}
    )

    assert response.status_code == 422