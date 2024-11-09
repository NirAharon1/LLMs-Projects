import os
from typing import List
import sqlite3

from langchain.indexes import SQLRecordManager, index
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv


INDEX_NAME = 'llamaparse-1536'
DATABASE_PATH = "nbt_pdf_database.db"
embeddings_model_name="text-embedding-3-small" # US$0.02 / 1M tokens

load_dotenv()
llama_cloud_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
embeddings_model = OpenAIEmbeddings(model=embeddings_model_name, openai_api_key=openai_api_key)


pc = Pinecone(api_key=pinecone_api_key)
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
            INDEX_NAME, 
            dimension=1536, 
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            ))
pc_index = pc.Index(INDEX_NAME)


embeddings_model = OpenAIEmbeddings(model=embeddings_model_name, openai_api_key=openai_api_key)
vectorstore = PineconeVectorStore(index=pc_index, embedding=embeddings_model)


# Create record manager
namespace = f"pinecone/{INDEX_NAME}"
record_manager = SQLRecordManager(
    namespace, db_url="sqlite:///record_manager_cache.sql"
)
record_manager.create_schema()


def get_db_connection():
    return sqlite3.connect(DATABASE_PATH)

def fetch_vectors_for_pinecone() -> List[Document]:
    docs = []
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Modified query to include date from pdfs table
        cursor.execute("""
            SELECT 
                v.name,
                v.file_name,
                v.page_number,
                v.parent_sha256,
                v.vector_sha256,
                v.markdown,
                v.embedding_openAI_small,
                v.last_update_time,
                p.title,
                p.date,
                p.url
            FROM vectors v
            LEFT JOIN pdfs p
                ON v.parent_sha256 = p.sha256
            ORDER BY v.name, v.page_number
        """)
        
        rows = cursor.fetchall()
        
        for row in rows:
            (name, file_name, page_number, parent_sha256, 
             vector_sha256, markdown, embedding_str, update_time,title, date, url) = row
            
            # Create metadata dictionary with date
            metadata = {
                "name": name,
                "file_name": file_name,
                "page_number": int(page_number),
                "parent_sha256": parent_sha256,
                "vector_sha256": vector_sha256,
                "source": f"{file_name}::{page_number}",
                "update_time": update_time,
                "title": title,
                "date": date,
                "pdf_url":url

            }
            
            # Create Langchain Document object
            doc = Document(
                page_content=markdown,
                metadata=metadata
            )
            docs.append(doc)
    
    return docs

docs = fetch_vectors_for_pinecone()

pinecone_update_result = index(
    docs,
    record_manager,
    vectorstore,
    cleanup="full"
)

print(pinecone_update_result)