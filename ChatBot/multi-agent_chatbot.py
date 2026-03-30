import os

from typing import TypedDict, Literal, Any
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate


class StateSchema(TypedDict):
    query: str
    choice: Literal["custom","smart"]
    route: Literal["tool","web","rag"]
    context: str
    answer: str
    llm: Any

def choice(state: StateSchema):
    state["llm"] = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key= os.getenv("OPENAI_API_KEY"))

    return state

## Rule based router agent
def custom_router(state: StateSchema):

    print("Custom Router Called...") 
    query = state["query"].lower()

    if any(word in query for word in['calculate','add','multiply','divide']):
        state["route"] = "tool"
    elif any(word in query for word in['amount','salary','policy']):
        state["route"] = "rag"
    elif any(word in query for word in['news','search','today']):
        state["route"] = "web"
    else:
        state["route"] = "web"

    return state

def smart_router(state: StateSchema):
    print("Smart Router Called...")
    prompt = f"""
            Classify query into:
                - tool: if it involves math, numbers, or calculation
                - rag: if it asks about salary, amount, or policy
                - web: if it asks for general knowledge, news, or search

                Query:
                {state['query']}
            """
    
    state["route"] = state["llm"].invoke(prompt).content.strip().lower()

    return state

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    clean_expr = expression.lower().replace("calculate", "").strip()
    return str(eval(clean_expr))

def tool_agent(state: StateSchema):
    print("Tool Agent Called...")
    state["answer"] = calculator.invoke(state["query"])

    return state

def web_agent(state: StateSchema):
    print("Web Agent Called...")
    response = state["llm"].invoke(state["query"])
    state["answer"] = response.content
    return state

def rag_agent(state: StateSchema):
    print("RAG Agent Called...")
    embeddings = OpenAIEmbeddings()
    chromaDB = Chroma(persist_directory="./ChromaDB",
                      collection_name="wipro",
                      embedding_function= embeddings)
    
    retriever = chromaDB.as_retriever(search_kwargs={"k":3})
    retrieved_docs = retriever.invoke(state["query"])
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = PromptTemplate(input_variables=["query","context"],
                            template="""
                                You are a strict RAG assistant.
                                Use only provided context to generate answer.
                                If answer is not present in the provided context say, 'I Don't Know'

                                Query:
                                {query}

                                Context:
                                {context}

                                Answer:
                            """)
    
    final_prompt = prompt.format(query= state["query"], context= context)

    response = state["llm"].invoke(final_prompt)
    state["answer"] = response.content

    return state


def answer(state: StateGraph):
    print("Final Answer Called...")
    return state

builder = StateGraph(StateSchema)


builder.add_node("choice",choice)
builder.add_node("custom_router",custom_router)
builder.add_node("smart_router",smart_router)
builder.add_node("tool_agent",tool_agent)
builder.add_node("web_agent",web_agent)
builder.add_node("rag_agent",rag_agent)
builder.add_node("answer",answer)


builder.add_edge(START,"choice")
builder.add_conditional_edges("choice", lambda state: "custom" if state["choice"] == "custom" else "smart",
                              {
                                  "custom":"custom_router",
                                  "smart": "smart_router"
                              })

builder.add_conditional_edges("custom_router", lambda state: 
                              "tool" if state["route"] == "tool"
                              else "rag" if state["route"] == "rag" 
                              else "web" if state["route"] == "web"
                              else "web",
                              {
                                  "tool": "tool_agent",
                                  "web": "web_agent",
                                  "rag": "rag_agent"
                              }
                              )
builder.add_conditional_edges("smart_router", lambda state: 
                              "tool" if state["route"] == "tool"
                              else "rag" if state["route"] == "rag" 
                              else "web" if state["route"] == "web"
                              else "web",
                              {
                                  "tool": "tool_agent",
                                  "web": "web_agent",
                                  "rag": "rag_agent"
                              }
                              )

builder.add_edge("tool_agent","answer")
builder.add_edge("web_agent","answer")
builder.add_edge("rag_agent","answer")
#builder.add_edge("smart_router","answer")
builder.add_edge("answer",END)
app = builder.compile()

"""
if __name__ == "__main__":
    result = app.invoke(
        {
            "query": input("Enter your query here: "),
            "choice": input("Enter your choice - ['custom','smart']: ")
        }
    )
    print(f"Final output: {result['answer']}")
"""
if __name__ == "__main__":
    print("Chatbot started. Type 'exit' to quit.")
    choice_mode = input("Enter your choice - ['custom','smart']: ")
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            print("Chatbot stopped.")
            break
        result = app.invoke({"query": query, "choice": choice_mode})
        print(f"Bot: {result['answer']}")
