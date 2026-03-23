import os

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from datasets import Dataset

import openai
from ragas.evaluation import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
#from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory

openai_client = openai.OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
ragas_llm = llm_factory("gpt-4o-mini", client=openai_client)
ragas_embeddings = embedding_factory(provider="openai", 
                                     model="text-embedding-3-small",
                                     client=openai_client,
                                     interface="modern"
                                     )

llm = ChatOpenAI(model="gpt-4o-mini",temperature=0,api_key= os.getenv("OPENAI_API_KEY"))
llm_embedding = OpenAIEmbeddings()
chromaDB = Chroma(persist_directory="./chroma_store",
                  collection_name="wipro",
                  embedding_function= llm_embedding)

## Data Ingestion
loader = PDFPlumberLoader(r"F:\GEN_AI\RAG\data\Offer_Letter.pdf")
loaded =loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
docs = splitter.split_documents(loaded)

chromaDB.add_documents(docs)

retriever = chromaDB.as_retriever(search_args={"k":3})

def ask_question(query):
    retriever_docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in retriever_docs ])

    prompt = PromptTemplate(input_variables=["query","context"],
                            template= """
                                    You are a AI assistant.
                                    Use only provided context to generate answer.
                                    If answer is not present in provided context say, 'I don't know'

                                    Query:
                                    {query}

                                    Context:
                                    {context}

                                    Answer:
                            """)
    
    final_prompt = prompt.format(query= query, context=context)
    response = llm.invoke(final_prompt)
    return response.content , context

if __name__ == "__main__":
    query = input("Enter Your Query here: ")
    answer, context = ask_question(query=query)

    print(f"AI Answer: \n {answer}")

    print("<<------- RAGAS Evaluation ------>>")

    dataset = Dataset.from_list([
                {
                "question": query,
                "contexts": [context],
                "answer": answer,
                "reference": answer
                }
                ])

    metrics = [
    faithfulness,
    #answer_relevancy,
    context_precision,
    context_recall
    ]

    """
    metrics = [
    Faithfulness(llm=ragas_llm),
    AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
    ContextPrecision(llm=ragas_llm, embeddings=ragas_embeddings),
    ContextRecall(llm=ragas_llm, embeddings=ragas_embeddings),
    ]
    """
    
    #print("\n==== METRIC DEBUG ====")
    #for i, m in enumerate(metrics):
    #    print(f"{i}: VALUE={m} | TYPE={type(m)}")
    

    os.environ["OPENAI_MAX_GENERATIONS"] = "1"
    ragas_result = evaluate(dataset, metrics=metrics)
    
    df = ragas_result.to_pandas()
    scores_df = df[['faithfulness', 'context_precision', 'context_recall']]
    print("\n📊 RAGAS Evaluation Scores:")
    print(scores_df.round(3).to_string(index=False))

