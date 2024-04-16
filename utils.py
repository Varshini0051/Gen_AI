import openai
import langchain
import pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import SpacyEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import openai
from pinecone import Pinecone, ServerlessSpec
import os   
from dotenv import load_dotenv
load_dotenv()

pc = Pinecone(api_key= os.getenv('pinecone_api_key'))
index_name= 'storydb'
# if index_name not in pc.list_indexes().names():
#     pc.create_index(
#         name=index_name, 
#         dimension=96, 
#         metric='cosine',
#         spec=ServerlessSpec(
#             cloud='aws',
#             region='us-west-2'
#         )
#     )
index = pc.Index(index_name)

# print(index.describe_index_stats())

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

pdf_path= 'document/sample.pdf'
doc= read_doc(pdf_path)
document_chunks= chunk_data(doc=doc)

# to convert the chunks to embeddings
# print(embeddings)

embeddings = SpacyEmbeddings(model_name="en_core_web_sm")

# Convert chunks to vectors
chunk_vectors = []
for chunk in document_chunks:
    chunk_text = chunk.page_content  # Extract text content from the chunk
    chunk_vectors.append(embeddings.embed_query(chunk_text))


# Insert vectors into the Pinecone index
index.upsert([(f"Chunk_{i}", vec) for i, vec in enumerate(chunk_vectors)])


# Perform similarity search
query = "knowledge and wisdom he had gained along the way, he set out to explore new worlds and "
query_vector = embeddings.embed_query(query)
print(len(query_vector))
similar_documents = index.query(vector=query_vector, top_k=5, namespace=index_name)
print(similar_documents)
# print(query_vector)