import json

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

load_dotenv()


# Connect to embedding model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# Connect to existing Chroma database
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="requirements"
)


# Load evaluation questions
with open("data/evaluation_questions.json", "r") as file:
    evaluation_data = json.load(file)


hit_at_1 = 0
hit_at_3 = 0


# Test every question
for test in evaluation_data:

    question = test["question"]
    expected_id = test["expected_requirement_id"]

    # Retrieve top 3 documents
    results = vectorstore.similarity_search_with_score(
        question,
        k=3
    )

    # Extract requirement IDs
    retrieved_ids = [
        doc.metadata["requirement_id"]
        for doc, distance in results
    ]

    print("\n----------------------------")
    print("Question:", question)
    print("Expected:", expected_id)
    print("Retrieved:", retrieved_ids)

    # Check Hit@1
    if retrieved_ids[0] == expected_id:
        hit_at_1 += 1

    # Check Hit@3
    if expected_id in retrieved_ids:
        hit_at_3 += 1


# Calculate final scores
total = len(evaluation_data)

hit_at_1_score = hit_at_1 / total
hit_at_3_score = hit_at_3 / total


print("\n============================")
print("EVALUATION RESULTS")
print("============================")

print(f"Total Questions: {total}")

print(
    f"Hit@1: {hit_at_1}/{total} "
    f"({hit_at_1_score:.2%})"
)

print(
    f"Hit@3: {hit_at_3}/{total} "
    f"({hit_at_3_score:.2%})"
)