import time
import os
from urllib.parse import urlparse, urljoin
import hashlib
import datetime
import sqlite3
from contextlib import contextmanager
import random
import re

import requests
from requests.exceptions import RequestException
from bs4 import BeautifulSoup
from tqdm import tqdm
from pypdf import PdfReader
from io import BytesIO


BASE_URL = "https://www.boi.org.il/roles/supervisionregulation/nbt/"
FOLDER_PATH = "nbt_pdf_folder"
DATABASE_PATH = "nbt_pdf_database.db"
MAX_RETRIES = 5
RETRY_DELAY = 1


class Pdf:
    def __init__(self, name, file_name, title, nbt_number, date, url, sha256, last_updated, num_pages):
        self.name = name
        self.file_name = file_name
        self.title = title
        self.nbt_number = nbt_number
        self.date = date
        self.url = url
        self.sha256 = sha256
        self.last_updated = last_updated
        self.num_pages =num_pages

    def __str__(self):
        return f"""Pdf name: {self.name}\n
                   PDF path: {self.file_name}\n
                   Title: {self.title}\n
                   nbt_number: {self.nbt_number}\n
                   Date: {self.date}\n
                   URL: {self.url}\nS
                   HA256: {self.sha256}\n
                   Last Updated: {self.last_updated} \n
                   Number of pages: {self.num_pages}"""
    

def get_pdf_data(url: str) -> list[Pdf]:
    """Scraping the base url, looking for pdf document of nbt"""
    pdf_data = []
    try:
        response = requests.get(url, timeout=20)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            for row in soup.find_all('tr'):
                title_elem = row.find('td')
                date_elem = row.find('td', {'data-order': True})
                link_elem = row.find('a')

                # Check if all elements are found
                if title_elem and link_elem:
                    title = title_elem.get_text(strip=True)
                    title_cleaned = re.sub(r'\\/:*?"<>|', '', title)
                    title_cleaned = title_cleaned.replace('"', '').replace("'", '').replace('/', '')
                    nbt_number = link_elem.get_text(strip=True)
                    pdf_url = urljoin(url, link_elem['href'])
                    date = date_elem.get_text(strip=True) if date_elem else None
                    name = f"{nbt_number} - {title_cleaned}"
                    file_name = f"{name}.pdf"

                    try:
                        response = requests.get(pdf_url, timeout=20)
                        if response.status_code == 200:
                            sha256 = hashlib.sha256(response.content).hexdigest()
                            last_updated = datetime.datetime.now()
                            pdf_content = BytesIO(response.content)  # Wrap content in BytesIO for PdfReader
                            pdf_reader = PdfReader(pdf_content)  # Load PDF content with PdfReader
                            num_pages = len(pdf_reader.pages)  # Get the number of pages
                            pdf = Pdf(name,file_name, title_cleaned, nbt_number, date, pdf_url, sha256, last_updated, num_pages)
                            pdf_data.append(pdf)
                    except RequestException as e:
                        print(f"Error fetching PDF from {pdf_url}: {str(e)}")
            return pdf_data
        else:
            print(f"Failed to fetch HTML from {url}. Status code: {response.status_code}")
            return []
    except RequestException as e:
        print(f"Network-related error occurred while fetching the URL: {str(e)}")
    except AttributeError as e:
        print(f"Parsing error: {str(e)} - Possibly due to unexpected HTML structure.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return []
    


def save_pdf(pdf: Pdf, path):
    try:
        # Check if the file already exists in the local folder
        file_path = os.path.join(path, pdf.file_name)
        if os.path.exists(file_path):
            # If file exists, calculate its hash
            with open(file_path, 'rb') as existing_file:
                existing_file_hash = hashlib.sha256(existing_file.read()).hexdigest()
            # If the hashes match, skip the file
            if existing_file_hash == pdf.sha256:
                print(f"File {pdf.file_name} already exists in the local folder with the same content. Skipping download.")
                return 'skipped'

        # Check if the file exists in the database
        result = check_pdf_in_db(pdf.file_name)
        if result:
            existing_sha256 = result[0]
            # If the file is in the database but not in the local folder, redownload it
            if not os.path.exists(file_path):
                print(f"File {pdf.file_name} exists in the database but not in the local folder. Redownloading...")
                response = requests.get(pdf.url, timeout=20)
                if response.status_code == 200:
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    update_pdf_in_db(pdf)
                    return 'redownloaded'
            # Overwrite only if content is different
            elif existing_sha256 != pdf.sha256:
                print(f"File {pdf.file_name} exists but has different content. Overwriting...")
                response = requests.get(pdf.url, timeout=20)
                if response.status_code == 200:
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    update_pdf_in_db(pdf)
                    print(f"PDF overwritten successfully: {file_path}")
                    return 'updated'
            else:
                print(f"File {pdf.file_name} already exists with the same content. Skipping download.")
                return 'skipped'
        else:
            # Save if file doesn't exist
            response = requests.get(pdf.url, timeout=20)
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f"PDF saved successfully: {file_path}")
                insert_pdf_in_db(pdf)
                return 'new'
            else:
                print(f"Failed to fetch PDF from {pdf.url}. Status code: {response.status_code}")
                return 'failed'
    except Exception as e:
        print(f"An error occurred while saving PDF: {str(e)}")
        return 'failed'
    

def get_pdf_sha256(content):
    """Calculate SHA256 hash of file content."""
    return hashlib.sha256(content).hexdigest()


# Create folder if it doesn't exist
if not os.path.exists(FOLDER_PATH):
    os.makedirs(FOLDER_PATH)
    print(f"Folder '{FOLDER_PATH}' created.")


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

def create_db_table():
    with get_db_connection() as conn:
        execute_with_retry(conn.execute, '''CREATE TABLE IF NOT EXISTS pdfs
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                         name TEXT,
                         file_name TEXT,
                         title TEXT,
                         nbt_number TEXT,
                         date TEXT,
                         url TEXT,
                         sha256 TEXT,
                         last_updated DATETIME,
                         num_pages INTEGER
                        )''')
        
def insert_pdf_in_db(pdf: Pdf):
    with get_db_connection() as conn:
        execute_with_retry(conn.execute,
            """INSERT INTO pdfs (name, file_name, title, nbt_number, date, url, sha256, last_updated, num_pages) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (pdf.name, pdf.file_name, pdf.title, pdf.nbt_number, pdf.date, pdf.url, pdf.sha256, pdf.last_updated, pdf.num_pages))

def update_pdf_in_db(pdf: Pdf):
    with get_db_connection() as conn:
        execute_with_retry(conn.execute,
            "UPDATE pdfs SET sha256 = ?, last_updated = ? WHERE file_name = ?",
            (pdf.sha256, pdf.last_updated, pdf.file_name))
    

def check_pdf_in_db(file_name: str):
    with get_db_connection() as conn:
        cursor = execute_with_retry(conn.execute,
            "SELECT sha256 FROM pdfs WHERE file_name = ?",
            (file_name,))
        return cursor.fetchone()
    

create_db_table()
pdf_data = get_pdf_data(BASE_URL)


start_time = time.time()
new_pdfs = 0
updated_pdfs = 0
failed_pdfs = 0
skipped_pdfs = 0
redownloaded_pdfs = 0


for pdf in tqdm(pdf_data, desc="Saving PDFs"):
    print(f"{pdf.file_name}  PDF URL {pdf.url}")
    result = save_pdf(pdf, FOLDER_PATH)
    if result == 'new':
        new_pdfs += 1
    elif result == 'updated':
        updated_pdfs += 1
    elif result == 'failed':
        failed_pdfs += 1
    elif result == 'skipped':
        skipped_pdfs += 1
    elif result == 'redownloaded':
        redownloaded_pdfs += 1

print(f"Total files saved successfully: {new_pdfs + updated_pdfs + redownloaded_pdfs} out of: {len(pdf_data)} urls  found in the website")
print(f"New PDFs: {new_pdfs}")
print(f"Updated PDFs: {updated_pdfs}")
print(f"Redownloaded PDFs: {redownloaded_pdfs}")
print(f"Failed PDFs: {failed_pdfs}")
print(f"Skipped PDFs: {skipped_pdfs}")
end_time = time.time()
print(f"Total time taken: {round(end_time - start_time, 2)} seconds")