import os
import json
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import PromptTemplate
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever

from flashrank import Ranker
from langchain_community.document_compressors import FlashrankRerank

from pydantic import BaseModel, Field
from typing import Type, List

class Citation(BaseModel):
    index: int
    source: str
    chunk_id : int
    page: int

class RAG_Response(BaseModel):
    query: str
    answer: str
    reference: List[Citation]

llm = ChatOpenAI(model="gpt-4o-mini",temperature=0, api_key= os.getenv("OPENAI_API_KEY"))
structured_llm = llm.with_structured_output(RAG_Response)

loader = PDFPlumberLoader("F:\GEN_AI\Graph_CrewAI\data\Offer_Letter.pdf")
loaded = loader.load()
print("PDF File Loaded..")

splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
docs = splitter.split_documents(loaded)

## For Citation
for i , doc in enumerate(docs):
    doc.metadata["source"] = "Wirpo_Offer"
    doc.metadata["chunk_id"] = i

print("Document splitted into chunks..")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
chromaDB = Chroma(persist_directory="./ChromaDB",
                  collection_name="ibm",
                  embedding_function=embeddings)

chromaDB.add_documents(docs)
print("Data loaded into vectorStore..")

## Hybrid Retriever
dense_retriever = chromaDB.as_retriever(search_kwargs={"k":10})
sparse_retriever = BM25Retriever.from_documents(docs)
sparse_retriever.k = 10

hybrid_retriever = EnsembleRetriever(retrievers=[dense_retriever,sparse_retriever],
                                     weights=[0.7,0.3]
                                     )


FlashrankRerank.model_rebuild()
ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

cross_encoder = FlashrankRerank(client=ranker,model="ms-marco-MiniLM-L-12-v2") # or JinaRerank
#cohera_rerank_retriever = CohereRerank(model="rerank-english-v2.0",cohere_api_key=???)

rerank_retriever = ContextualCompressionRetriever(base_retriever = hybrid_retriever,
                                                  base_compressor = cross_encoder
                                                  )


def ask_question(query: str) -> str:

    retrieved_docs = rerank_retriever.invoke(query)
    context_parts = []

    for i, doc in enumerate(retrieved_docs):
        source = doc.metadata.get("source","unknown")
        chunk_id = doc.metadata.get("chunk_id",i)
        page = doc.metadata.get("page", None)
        if page is not None:
            page = page + 1

        context_parts.append(f"[{i}] (Source:{source}, Chunk:{chunk_id}, Page: {page})\n{doc.page_content}")

    context = "\n\n".join(context_parts)

    prompt = PromptTemplate(input_variables=["query","context"],
                            template= """
                                You are a strict RAG assistant.
                                
                                Rule:
                                    - Use only provided context to generate answer.
                                    - If answer is not present in the provided context say, 'I don't have information'
                                    - Add citation in answer
                                    - Only include citations that are actually used
                                
                                Return structured output.

                                Question:
                                {query}

                                Context:
                                {context}

                                Answer (with citation):                            
                            """)
    
    final_prompt = prompt.format(query= query, context= context)

    response = structured_llm.invoke(final_prompt)
    
    return response

if __name__ == "__main__":
    query = input("Enter your query here: ")
    result = ask_question(query)

    print(json.dumps(result.model_dump(), indent=4))
    #print(f"Final Answer: \n{result.answer} \n")

    
    #print("Citation: ")
    #for c in result.citation:
    #    print(f"[{c.index}] Source: {c.source}, Chunk: {c.chunk_id}, Page: {c.page}")

