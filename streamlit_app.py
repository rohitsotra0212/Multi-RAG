import streamlit as st
from PDFLoader.PDFLoader_main import pdf_main   # Import the function from main.py
from TextLoader.TextLoader_main import text_main

st.set_page_config(page_title="Multi Loader App", layout="wide")

st.title("Welcome to Multi Loader Test")

# Sidebar navigation (persistent)
st.sidebar.title("Choose Loader")
loader_choice = st.sidebar.radio("Select an option:", ["Home", "TextLoader", "PDFLoader", "WebBaseLoader"])

# Home Screen
if loader_choice == "Home":
    st.info("👋 Select a loader from the sidebar to continue.")

# TextLoader placeholder
elif loader_choice == "TextLoader":
    st.success("Text Loader functionality will go here.")
    text_main()

# PDF Loader -> Call main.py function
elif loader_choice == "PDFLoader":
    st.success("PDF Loader functionality will go here.")
    pdf_main()   # Executes the full PDF app inside this tab

# Web Loader placeholder
elif loader_choice == "WebBaseLoader":
    st.success("Web Loader functionality will go here.")
