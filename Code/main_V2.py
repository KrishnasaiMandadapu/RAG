import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from uuid import uuid4
import streamlit as st
import chromadb
from langchain_chroma import Chroma
from PIL import Image
from backend import *

def get_vector_db_details():
   
    client = chromadb.PersistentClient(path="D:/Python_project")
    collection = client.get_collection(name="test") 
    db_details = {
        "Database Name": "Chroma Vector DataBase",
        "Collection Name": "test",
        "Number of Chunks": collection.count()
    }
    return db_details

# Sidebar Section
def load_sidebar():
    
    st.sidebar.image("D:/Python_project/txst.jpg", use_container_width=True)  # Replace with your logo path

    st.sidebar.title("📊 Vector Database Info")
    db_details = get_vector_db_details()
    for key, value in db_details.items():
        st.sidebar.write(f"**{key}:** {value}")

    st.sidebar.info("Ensure the database is active and connected.")

def upload_files(uploaded_files=[]):
    """Upload files and prevent duplicates."""
    st.title("📁 File Upload & Search Interface")
    new_uploaded_files = st.file_uploader(
        "Upload your files (DOCX, CSV, PDF):", 
        type=["docx", "csv", "pdf"], 
        accept_multiple_files=True
    )
    
    
    if new_uploaded_files:
        for file in new_uploaded_files:
            if file.name not in [f.name for f in uploaded_files]: 
                uploaded_files.append(file)
                
            else:
                st.warning(f"Duplicate file: {file.name}. This file has already been uploaded.")
    
    return uploaded_files

# Display Uploaded Files
def display_uploaded_files(files):
    if files:
        st.subheader("Uploaded Files")
        for file in files:
            st.write(f"📄 {file.name}")
    else:
        st.info("No files uploaded yet.")

# File Processing Placeholder
def process_files(obj,files):
    
    with st.spinner("Processing files..."):
        
        
        obj.process_uploaded_files(files)
        # st.success("Files processed successfully!")

# Search Bar Section
def search_interface(obj):
    st.subheader("🔍 Search the Processed Data")
    query = st.text_input("Enter your query:")
    if st.button("Search"):
        if query:
            
            result=obj.get_docs_summary(query)
            st.info(f"Results for '{query}':")
            st.write(result['output_text'])
            st.subheader("Retrieved Chunks and Sources:")
            for idx, item in enumerate(result['input_documents'], start=1):
                st.write(f"**Chunk {idx}:** {item.page_content}")
                st.write(f"**Source:** {item.metadata['source']}")
                st.divider() 
        else:
            st.error("Please enter a query to search.")


def main():
    
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    
    st.set_page_config(layout="wide")
    obj = Rag()
    st.markdown("<h2 style='text-align: center; color: rgb(227, 38, 54);'>Retrieval Augmented Generation</h2>", unsafe_allow_html=True)
    load_sidebar()
    
    
    new_uploaded_files = upload_files()  
    display_uploaded_files(new_uploaded_files)
    
   
    if new_uploaded_files:
        
        new_files_to_process = [
            file for file in new_uploaded_files
            if file.name not in [f.name for f in st.session_state.uploaded_files]
        ]
        
        if st.button("Process Files"):  
            if new_files_to_process:
                
                process_files(obj, new_files_to_process)
                
                st.session_state.uploaded_files.extend(new_files_to_process)
                st.success("Processing complete! Newly uploaded files have been processed.")
            else:
                st.warning("No new files to process.")
            

    search_interface(obj)

if __name__ == "__main__":
    main()




