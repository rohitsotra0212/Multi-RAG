import os

from dotenv import load_env
load_env()

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from lanchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings()
chromaDB = Chroma(persist_directory="./chromaDB",
                  collection_name="wipro_internal",
                  embedding_function= embeddings
                 )

loader = PyMuPDFLoader("<input_file.pdf>")
loaded = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
docs = splitter.split_documents(docs)

dense_retriever = chromaDB.as_retriever(search_kwargs={"k":8})
sparse_retriever = BM25Retriever.from_documents(docs)
sparse_retriever.k = 8

## Hybrid Retriever
hybrid_retriever = EnsembleRetriever(retrievers = [dense_retriever, sparse_retriever],
                                     weights=[0.7,0.3]
                                    )

def ask_question(query: str) -> str:
  retrieved_docs = hybrid_retriever.invoke(query)
  context = "\n\n.join([doc.page_content doc in retrieved_docs])

  prompt = PromptTemplate(input_variables=["query","context"],
                        template="""
                        You are a AI Assistant.
                        Use only provided context to generate answer.
                        If answer is not present in the provided context say, I Don't know.

                        Query:
                        {query}

                        Context:
                        {context}

                        Answer:
                        """
                       )

  final_prompt = prompt.format(query= query, context= context)

  response = llm.invoke(final_prompt)
  return response.context

if __name__ == "__main__":
  query = input("Enter your query here: ")
  result = ask_question(query)

  print(f" AI Output: \n {result})












