import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma 
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate

## Data Load
loader = PyMuPDFLoader("F:\GEN_AI\RAG\data\Offer_Letter.pdf")
loaded = loader.load()

## Split into Chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
docs = splitter.split_documents(loaded)

## Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

## Store in Vector Stores "Chroma"
chromaDB = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="wipro_offer"
)

chromaDB.add_documents(docs)

## Create 
prompt = PromptTemplate(
    input_variables=["context","question"],
    template=""" You are an AI assistant.
    Use only provided context to generate answers.
    If answer is not present in the context, say "Not Found in the PDF".

    Context:
    {context}

    Question:
    {question}

    Answer:
"""
)

## Create OpenAI LLM
load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini",
                temperature=0,
                api_key=os.getenv("OPENAI_API_KEY"))

## Create Retrieval
retriever = chromaDB.as_retriever(search_kwargs={"k":3})

## Query Function
def ask_question(query):
    retrieved_docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    final_prompt = prompt.format(
        context= context,
        question= query
    )

    response = llm.invoke(final_prompt)
    return response.content

if __name__ == "__main__":
    query = "What is Gross Transport recovery amount?"
    answer = ask_question(query)

    print("Final Output: \n ")
    print(answer)
