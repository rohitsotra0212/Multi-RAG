import os
import json
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import LLMChain

# -------------------
# Streamlit App
# -------------------
st.set_page_config(page_title="PFD Extractor", layout="wide")
st.title("📑 PDF Extractor - Information")

# Sidebar - API Key
openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

# File Upload
uploaded_file = st.file_uploader("Upload a Payslip PDF", type=["pdf"])

# Output path
output_path = st.sidebar.text_input("Enter path to save JSON (e.g., ./output.json)", "./output.json")

if uploaded_file and openai_api_key:
    with st.spinner("Processing PDF..."):

        # Save file temporarily
        save_dir = r"F:\Langchain\PromptTemplate\temp"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 1. Load PDF
        loader = PyPDFLoader(save_path)
        raw_docs = loader.load()
        
        # 2. Split text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(raw_docs)

        # 3. Embed and store in FAISS
        embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        vectorstore = FAISS.from_documents(docs, embeddings)

        st.success("✅ PDF processed, embeddings stored in FAISS") 

        # 4. Load prompt from external file (prompt.md)
        with open("F:\Langchain\PromptTemplate\prompt\prompt.md", "r", encoding="utf-8") as f:
            template = f.read()

        prompt = PromptTemplate(input_variables=["context"], template=template)

        llm = ChatOpenAI(api_key=openai_api_key, temperature=0, model="gpt-3.5-turbo")
        chain = LLMChain(llm=llm, prompt=prompt)

        # Combine all PDF text
        full_text = "\n".join([d.page_content for d in docs])

        result = chain.run({"context": full_text})

        try:
            data = json.loads(result)
        except:
            st.error("❌ Could not parse JSON. Here’s the raw output:")
            st.text(result)
            data = None

        if data:
            st.json(data)

            # Save to JSON file
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)

            st.success(f"✅ JSON saved to {output_path}")        