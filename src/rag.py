from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma


load_dotenv()


# -------------------------
# EMBEDDING MODEL
# -------------------------

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# -------------------------
# VECTOR DATABASE
# -------------------------

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="requirements"
)


# -------------------------
# LLM
# -------------------------

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)


# -------------------------
# RAG FUNCTION
# -------------------------

def ask_rag(question):

    # 1. Retrieve documents
    results = vectorstore.similarity_search_with_score(
        question,
        k=5
    )


    # 2. Prepare candidates for reranker
    candidate_text = ""

    for i, (doc, distance) in enumerate(results):

        candidate_text += f"""
DOCUMENT {i}

Requirement ID: {doc.metadata['requirement_id']}
Content: {doc.page_content}

"""


    # 3. Rerank documents
    rerank_prompt = f"""
You are a document relevance evaluator.

Given a user question and a list of retrieved documents,
rank the documents from most relevant to least relevant.

Return ONLY the document numbers separated by commas.

QUESTION:
{question}

DOCUMENTS:
{candidate_text}
"""


    rerank_response = llm.invoke(rerank_prompt)


    # Convert "0,1,2,3,4"
    # into [0, 1, 2, 3, 4]

    ranking = [
        int(x.strip())
        for x in rerank_response.content.split(",")
    ]


    # 4. Select top 3
    top_indices = ranking[:3]

    reranked_docs = [
        results[i][0]
        for i in top_indices
    ]


    # 5. Build context
    context = "\n\n".join(
        f"""
Requirement ID: {doc.metadata['requirement_id']}
Content: {doc.page_content}
"""
        for doc in reranked_docs
    )


    # 6. Final RAG prompt
    final_prompt = f"""
You are a software requirements assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context, say:
"I don't know based on the provided documentation."

Include the Requirement ID that supports your answer.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""


    # 7. Generate final answer
    response = llm.invoke(final_prompt)


    # 8. Return answer
    return response.content



if __name__ == "__main__":

    question = "How long does an account stay locked?"

    answer = ask_rag(question)

    print("\nFINAL RAG ANSWER:")
    print(answer)