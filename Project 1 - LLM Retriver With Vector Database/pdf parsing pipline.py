import time
import os
import hashlib
import sqlite3

from llama_parse import LlamaParse
from dotenv import load_dotenv
from langchain_openai.embeddings.base import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from contextlib import contextmanager
import random
import re

embeddings_model_name="text-embedding-3-small" # US$0.02 / 1M tokens
# embeddings_model_name="text-embedding-ada-002" # US$0.10 / 1M tokens

load_dotenv()
llama_cloud_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
embeddings_model = OpenAIEmbeddings(model=embeddings_model_name, openai_api_key=openai_api_key)
FOLDER_PATH = "nbt_pdf_folder"
DATABASE_PATH = "nbt_pdf_database.db"
MAX_RETRIES = 5
RETRY_DELAY = 1



    
def llama_parse_to_list(pdf_path: str) -> str:
    parser = LlamaParse(
        result_type="markdown",
        api_key=llama_cloud_api_key,
        show_progress=False,
    )
    return parser.load_data(pdf_path)


def increment_heading_level(text: str) -> str:
    lines = text.split('\n')
    result = []
    for line in lines:
        if line.startswith('###'):
            line = '##' + line  # Add one more '##'
        elif line.startswith('##'):
            line = '##' + line  # Add one more '##'
        elif line.startswith('#'):
            line = '##' + line  # Add one more '##'
        result.append(line)  # Add this line to append the modified line
    return '\n'.join(result)


def clean_text(text: str) -> str:
    # Define patterns to remove in a list of tuples (pattern, flags)
    patterns_to_remove = [
        (r"^# ניהול בנקאי תקין: המפקח על הבנקים\n?", 0),
        (r"^# ניהול בנקאי תקין\n?", 0),
        (r"^# המפקח על הבנקים", re.MULTILINE),
        (r"^# הפיקוח על הבנקים", re.MULTILINE),
        (r"^\s*\n", re.MULTILINE)
    ]
    
    # Apply all cleaning patterns
    cleaned_text = text
    for pattern, flags in patterns_to_remove:
        cleaned_text = re.sub(pattern, "", cleaned_text, flags=flags)
    
    cleaned_text = increment_heading_level(cleaned_text)
    return cleaned_text.strip()

def calculate_sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_PATH, timeout=20)  # Added timeout
        yield conn
    finally:
        if conn:
            conn.commit()  # Ensure all changes are committed
            conn.close()
def execute_with_retry(func, *args, **kwargs):
    """Execute a database operation with retry logic"""
    for attempt in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) and attempt < MAX_RETRIES - 1:
                sleep_time = RETRY_DELAY * (1 + random.random())  # Add some randomness to prevent deadlocks
                print(f"Database is locked, retrying in {sleep_time:.2f} seconds... (Attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(sleep_time)
            else:
                raise
        except Exception as e:
            print(f"Unexpected error in database operation: {str(e)}")
            raise

with get_db_connection() as conn:
    execute_with_retry(conn.execute, """
    CREATE TABLE IF NOT EXISTS vectors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        file_name TEXT,
        page_number INTEGER,
        parent_sha256 TEXT,
        vector_sha256 TEXT,
        markdown TEXT,
        embedding_openAI_small TEXT,
        last_update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")
    

conn = sqlite3.connect(DATABASE_PATH)
cursor = conn.cursor()

# Execute the query to get the top 5 records ordered by 'bnt_number'
cursor.execute("""
    SELECT 
        pdfs.name,
        pdfs.file_name,
        pdfs.sha256,
        parent_sha256
    FROM pdfs
    LEFT JOIN vectors
        on pdfs.sha256 = vectors.parent_sha256
    WHERE
        (vectors.parent_sha256 IS NULL)
        or
        (pdfs.name = vectors.name
         and pdfs.sha256 <> vectors.parent_sha256)
        and pdfs.name <> '368 - יישום תקן של בנקאות פתוחה בישראל'
    ORDER BY nbt_number
    LIMIT 100
""")

# Fetch and process the results
records_to_process = cursor.fetchall()
for (name, file_name, sha256, existing_parent_sha256) in records_to_process:
    print(f"Processing file: {name} ({file_name}) - SHA256: {sha256}")
    relative_path = os.path.join(FOLDER_PATH, file_name)
    
    # If this is an update to an existing file, delete old vectors
    if existing_parent_sha256:
        print("Updating existing file - removing old vectors")
        execute_with_retry(
            cursor.execute,
            "DELETE FROM vectors WHERE parent_sha256 = ?",
            (existing_parent_sha256,)
        )
        conn.commit()
    
    # Process the file
    llma_parse_result = llama_parse_to_list(relative_path)
    
    # Process each page and update the vectors table
    for page_number, page in enumerate(llma_parse_result, 1):
        # Generate embedding
        page_content_cleaned = clean_text(page.text)
        openai_embedding = embeddings_model.embed_documents([page_content_cleaned])[0]
        
        # Calculate vector SHA256
        vector_sha256 = calculate_sha256(str(openai_embedding))

        # Prepare the data for insertion
        vector_data = (
            name,
            file_name,
            page_number,
            sha256,  # parent_sha256
            vector_sha256,
            page_content_cleaned,  # markdown
            str(openai_embedding),  # embedding as string
        )
        
        # Insert the vector data into the database
        execute_with_retry(
            cursor.execute,
            """
            INSERT INTO vectors 
            (name, file_name, page_number, parent_sha256, vector_sha256, markdown, embedding_openAI_small)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            vector_data
        )
        
        # Commit after each page to ensure data is saved
        conn.commit()

# Close the database connection
conn.close()