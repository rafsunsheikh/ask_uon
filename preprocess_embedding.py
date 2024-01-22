import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os 
import requests
from langchain_community.document_loaders import WebBaseLoader


# Set the links for each FAISS database
equity_diversion_inclusion_link = ["https://www.newcastle.edu.au/our-uni/equity-diversity-inclusion", 
        "https://www.newcastle.edu.au/our-uni/equity-diversity-inclusion/equity-cohorts",
        "https://www.newcastle.edu.au/our-uni/equity-diversity-inclusion/about",
        "https://www.newcastle.edu.au/our-uni/equity-diversity-inclusion/gender-equity"]

home_phone_link = ["https://www.newcastle.edu.au/our-uni/equity-diversity-inclusion/inclusive-leadership"]

mobile_phone_link = ["https://www.newcastle.edu.au/our-uni/equity-diversity-inclusion/cultural-diversity-belonging"]


# Load the html text from the website
equity_diversion_inclusion_loader = WebBaseLoader(equity_diversion_inclusion_link)
equity_diversion_inclusion_data = equity_diversion_inclusion_loader.load()

home_phone_loader = WebBaseLoader(home_phone_link)
home_phone_data = home_phone_loader.load()

mobile_phone_loader = WebBaseLoader(mobile_phone_link)
mobile_phone_data = mobile_phone_loader.load()

# Split the html text into chunks
splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
equity_diversion_inclusion_chunks = splitter.split_documents(equity_diversion_inclusion_data)
home_phone_chunks = splitter.split_documents(home_phone_data)
mobile_phone_chunks = splitter.split_documents(mobile_phone_data)

# Create the FAISS database
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'})

equity_diversion_inclusion_db = FAISS.from_documents(equity_diversion_inclusion_chunks, embeddings)
home_phone_db = FAISS.from_documents(home_phone_chunks, embeddings)
mobile_phone_db = FAISS.from_documents(mobile_phone_chunks, embeddings)

# Save the FAISS database
equity_diversion_inclusion_db.save_local("equity_diversion_inclusion_faiss")
home_phone_db.save_local("home_phone_faiss")
mobile_phone_db.save_local("mobile_phone_faiss")
