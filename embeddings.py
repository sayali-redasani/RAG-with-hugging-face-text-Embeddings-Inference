import requests
## necessary imports
import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import pinecone
import string
import random
import numpy as np

#Read the text document
txt_loader = DirectoryLoader(os.getcwd(), glob="original_copy.txt")
documents = txt_loader.load()
print(documents)

# Split documents text to create chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500) #chunk overlap seems to work better
documents = text_splitter.split_documents(documents)
print(len(documents))

def custom_embedding_function(documents):
    """
    Generates embeddings for the given document using the deployed model.
    Input: A list of documents (documents)
    Processing: Iterates through the list of documents and calls the API for each document
    Output: A list of embedding vectors (response_embeddings)
    """
    API_URL = "http://192.168.1.125:8181/"
    embeddings_list = []
 
    for doc in documents:
        template = """Generate the Embeddings for the given document {doc} and return in the form of embed_documents type"""
        prompt = PromptTemplate.from_template(template=template).format(doc=doc)
        data = {
            "inputs": prompt,
        }
 
        response = requests.post(API_URL, json=data)
        response_embeddings = response.json()
        embeddings_list.extend(response_embeddings)  # Extend instead of append
 
    return embeddings_list

def custom_embedding_functionQ(doc):
    """
    Generates embeddings for the given document using the deployed model.
    Input: A list of documents (documents)
    Processing: Iterates through the list of documents and calls the API for each document
    Output: A list of embedding vectors (response_embeddings)
    """
    API_URL = "http://192.168.1.125:8181/"
    embeddings_list = []
 
    template = """Generate the Embeddings for the given document {doc} and return in the form of embed_documents type"""
    prompt = PromptTemplate.from_template(template=template).format(doc=doc)
    data = {
        "inputs": prompt,
    }

    response = requests.post(API_URL, json=data)
    response_embeddings = response.json()
    embeddings_list.extend(response_embeddings)  # Extend instead of append
    return embeddings_list

# Sending the retrieval documents to the text-generation model for proper answer to the query
def query_answer(query, passage):
    prompt = f"""
            Answer the query by considering the context
                
            {query}

            Context : {passage}                
           Answer : """
    
      
    url = "http://192.168.1.125:8081/generate"
    data = {
        "inputs": prompt,
        "parameters" :{"max_new_tokens":250}
    }
    response = requests.post(url=url, json=data)
    ans = response.json()["generated_text"]
    print(f"AI Bot: {ans}")

#Creating embeddings to the document provided
embeddings =  custom_embedding_function(documents)
embeddings = np.array(embeddings)
print(len(embeddings))

query = "What did Raskolnikov find?"
query_embedding = custom_embedding_functionQ(query)
query_embedding = np.array(query_embedding)

ids = [''.join(random.choices(string.ascii_uppercase, k=3)) for _ in range(len(documents))]
print(ids)

# Connect to Pinecone
PINECONE_API_KEY = "YOUR PINECODE API-KEY"
PINECONE_ENV = "gcp-starter"
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Upsert embeddings into Pinecone index
index_name = pinecone.Index("rag")
metadata = [{'chunks': doc.page_content} for doc in documents]

print(len(metadata))

# First time if you are storing the vectors into vectorDB
k = index_name.upsert(vectors=zip(ids, embeddings.tolist(),metadata))

#Search for matches
result = index_name.query(query_embedding.tolist(), top_k=3, include_metadata=True)
#print([result.matches[i].metadata for i in range(3)])

passage = " ".join([result.matches[i].metadata['chunks'] for i in range(3)])
print(passage)
query_answer(query , passage)
