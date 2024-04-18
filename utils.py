import openai
import langchain
import pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SpacyEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import openai
from pinecone import Pinecone, ServerlessSpec
import os   
from dotenv import load_dotenv
load_dotenv()

pc = Pinecone(api_key= os.getenv('pinecone_api_key'))
index_name= 'story'
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name, 
        dimension=96, 
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-west-2'
        )
    )
index = pc.Index(host='https://sample-uj69pc1.svc.apw5-4e34-81fa.pinecone.io')

# to read the document
def read_doc(directory):
    loader= PyPDFLoader(directory)
    documents= loader.load()
    return documents

# to divide the docs into chunks.
def chunk_data(doc, chunk_size=800, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    document= text_splitter.split_documents(doc)
    return document

# to perform similarity search.
def similarity_search(query, k):
    query_vector = embeddings.embed_query(query)
    similar_documents = index.query(vector=[query_vector], top_k=k, include_metadata=True, include_values=False)
    similar_id = similar_documents['matches'][0]['id']
    return(similar_documents,similar_id)


pdf_path= 'document/story.pdf'
document_name = os.path.basename(pdf_path)
doc= read_doc(pdf_path)
document_chunks= chunk_data(doc=doc)

embeddings = SpacyEmbeddings(model_name="en_core_web_sm")

# Convert chunks to vectors
chunk_data = []

for i, chunk in enumerate(document_chunks):
    chunk_text = chunk.page_content
    chunk_metadata = chunk.metadata
    chunk_metadata['text'] = chunk_text
    chunk_metadata['source']= document_name
    chunk_vector = embeddings.embed_query(chunk_text)
    
    chunk_data.append((f"{i}", chunk_vector, chunk_metadata))

# Insert vectors into the Pinecone index
index.upsert(chunk_data)

# Perform similarity search
query = "puzzled the pirates"
result= similarity_search(query,2)
print(result)