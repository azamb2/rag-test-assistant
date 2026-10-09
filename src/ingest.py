import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
import re


load_dotenv()


# Location of our requirements document
file_path = "data/software_requirements.txt"

# Create a loader
loader = TextLoader(file_path)

# Load the file into LangChain
documents = loader.load()

text = documents[0].page_content

pattern = r"((?:AUTH|CHECKOUT|API)-\d{3}):\s*(.*?)(?=\n(?:AUTH|CHECKOUT|API)-\d{3}:|\Z)"

matches = re.findall(
    pattern,
    text,
    flags=re.DOTALL
)

chunks = []

for requirement_id, content in matches:

    content = content.strip()

    content = content.replace("Checkout Requirements", "")
    content = content.replace("API Requirements", "")
    content = content.strip()

	
    category = requirement_id.split("-")[0]

    chunk = Document(
        page_content=content,
        metadata={
            "requirement_id": requirement_id,
            "category": category,
            "source": "software_requirements.txt"
        }
    )

    chunks.append(chunk)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="requirements"
)

for chunk in chunks:
    print("\nCONTENT:")
    print(chunk.page_content)

    print("METADATA:")
    print(chunk.metadata)



print("Vector database created successfully.")
print("Chunks stored:", len(chunks))
