import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from uuid import uuid4
from langchain.chains.question_answering import load_qa_chain
import tempfile



class Rag:
    def __init__(self):
        
        os.environ['google_api_key'] = 'API_KEY'
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",google_api_key=os.getenv('google_api_key'))
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=os.getenv('google_api_key'))
    
        self.vector_store = Chroma(
            collection_name="test",
            embedding_function=self.embeddings,
            persist_directory="D:/Python_project",  
        )
        self.retriever = self.vector_store.as_retriever(search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 5, "lambda_mult": 0.5},
        )
    def handle_uploaded_files(self,uploaded_files):
        temp_file_paths = []

        # Save uploaded files temporarily and store their paths
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())  # Save file content to temp file
                temp_file_paths.append(temp_file.name)

        return temp_file_paths
    
    def process_pdf_files(self, paths):
        
        pdf_chunks=[]
        for path in paths:
            
            loader = PyPDFLoader(path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
            page_splits = text_splitter.split_documents(docs)
            for i, chunk in enumerate(page_splits, start=1):
                chunk.metadata['id'] = i
            pdf_chunks.extend(page_splits)

        return pdf_chunks
    
    def process_doc_files(self, paths):

        doc_chunks=[]
        for path in paths:
            loader = Docx2txtLoader(path)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
            docs=loader.load()
            page_splits = text_splitter.split_documents(docs)
            for i, chunk in enumerate(page_splits, start=1):
                chunk.metadata['id'] = i
            doc_chunks.extend(page_splits)

        return doc_chunks
    
    def process_csv_files(self, paths):

        csv_chunks=[]
        for path in paths:
            loader = CSVLoader(path)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
            docs=loader.load()
            page_splits = text_splitter.split_documents(docs)
            for i, chunk in enumerate(page_splits, start=1):
                chunk.metadata['id'] = i
            csv_chunks.extend(page_splits)

        return csv_chunks
    
    def process_uploaded_files(self, files):
        temp_file_paths = {"pdf": [], "csv": [], "docx": []}

        for uploaded_file in files:
            _, ext = os.path.splitext(uploaded_file.name)
            ext = ext.lower()

            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
                temp_file.write(uploaded_file.read())
                if ext == ".pdf":
                    temp_file_paths["pdf"].append(temp_file.name)
                elif ext == ".csv":
                    temp_file_paths["csv"].append(temp_file.name)
                elif ext == ".docx":
                    temp_file_paths["docx"].append(temp_file.name)
                  
            pdf_chunks = self.process_pdf_files(temp_file_paths["pdf"])
            csv_chunks = self.process_csv_files(temp_file_paths["csv"])
            docx_chunks = self.process_doc_files(temp_file_paths["docx"])
            if pdf_chunks:
                print("PDFFF",len(pdf_chunks))
                self.add_all_chunks_db(pdf_chunks)
            if csv_chunks:
                print("CSVVV",len(csv_chunks))
                self.add_all_chunks_db(csv_chunks)
            if docx_chunks:
                print("DOCC",len(docx_chunks))
                self.add_all_chunks_db(docx_chunks)

    def add_all_chunks_db(self, data):

        uuids = [str(uuid4()) for _ in range(len(data))]

        self.vector_store.add_documents(documents=data, ids=uuids)
        
    def get_docs_summary(self, query):

        docs=self.retriever.invoke(query)
        chain = load_qa_chain(llm=self.llm, chain_type="stuff")
       
        result=chain.invoke({"input_documents":docs, "question":[query]})
        return result






    
