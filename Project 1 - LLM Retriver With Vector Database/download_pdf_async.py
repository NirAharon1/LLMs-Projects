import asyncio
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import os
import sys
import time
import aiofiles
import json
import aiohttp


script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)

async def fetch_html(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

async def get_pdf_data(url):
    try:
        html = await fetch_html(url)
        soup = BeautifulSoup(html, 'html.parser')
        pdf_data = {}
        for row in soup.find_all('tr'):
            title_elem = row.find('td')
            date_elem = row.find('td', {'data-order': True})
            link_elem = row.find('a')

            # Check if title and url like elements are found
            if title_elem and link_elem:
                title = title_elem.get_text(strip=True)
                file_name = link_elem.get_text(strip=True)
                pdf_url = urljoin(url, link_elem['href'])
                if date_elem:
                    date = date_elem.get_text(strip=True)
                else:
                    date = None

                # Use the file name as the key for the dictionary
                pdf_data[file_name+".pdf"] = {
                    'title': title,
                    'date': date,
                    'pdf_url': pdf_url
                }
        return pdf_data
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return {}

async def save_pdf(file_name, pdf_url, folder_path):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(pdf_url) as response:
                if response.status == 200:
                    file_path = os.path.join(folder_path, file_name)
                    async with aiofiles.open(file_path, 'wb') as f:
                        await f.write(await response.read())
                    print(f"PDF saved successfully: {file_path}")
                else:
                    print(f"Failed to fetch PDF from {pdf_url}. Status code: {response.status}")
    except Exception as e:
        print(f"An error occurred while saving PDF: {str(e)}")

def retry_decorator(func):
    async def wrapper(*args, **kwargs):
        retries = 5
        delay = 3
        for attempt in range(retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < retries - 1:
                    print(f"Retrying in {delay} second(s)...")
                    await asyncio.sleep(delay)
        print("Failed after retries.")
    return wrapper

@retry_decorator
async def save_pdf_with_retry(file_name, pdf_url, folder_path):
    await save_pdf(file_name, pdf_url, folder_path)

async def main():
    start_time = time.time()
    base_url = "https://www.boi.org.il/roles/supervisionregulation/nbt/"
    folder_path = "pdf_folder"
    
    # Create folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    
    pdf_data = await get_pdf_data(base_url)
    tasks = []
    for file_name, pdf_info in tqdm(pdf_data.items(), desc="Saving PDFs"):
        pdf_url = pdf_info['pdf_url']
        tasks.append(save_pdf_with_retry(file_name, pdf_url, folder_path))
    await asyncio.gather(*tasks)
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")

    with open('pdf_data.json', 'w', encoding='utf-8') as json_file:
        json.dump(pdf_data, json_file, ensure_ascii=False)

if __name__ == "__main__":
    asyncio.run(main())


