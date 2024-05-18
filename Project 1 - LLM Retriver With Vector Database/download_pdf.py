import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import os
import json
from tqdm import tqdm
import time


base_url = "https://www.boi.org.il/roles/supervisionregulation/nbt/"
folder_path = "pdf_folder"

def get_pdf_data(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            pdf_data = {}
            for row in soup.find_all('tr'):
                title_elem = row.find('td')
                date_elem = row.find('td', {'data-order': True})
                link_elem = row.find('a')

                # Check if all elements are found
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
        else:
            print(f"Failed to fetch HTML from {url}. Status code: {response.status_code}")
            return []
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return []
    

def save_pdf(file_name, pdf_url, folder_path):
    try:
        response = requests.get(pdf_url)
        if response.status_code == 200:
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"PDF saved successfully: {file_path}")
        else:
            print(f"Failed to fetch PDF from {pdf_url}. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred while saving PDF: {str(e)}")


# Create folder if it doesn't exist
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created.")




if __name__ == "__main__":
    start_time = time.time()
    successful_saves = 0
    pdf_data = get_pdf_data(base_url)
    for file_name, pdf_info in tqdm(pdf_data.items(), desc="Saving PDFs"):
        pdf_url = pdf_info['pdf_url']
        print(file_name+" PDF URL:", pdf_url)
        if save_pdf(file_name, pdf_url, folder_path):
            successful_saves += 1
    print(f"Total files saved successfully: {successful_saves} out of: {len(pdf_data.items())} urls  found in the website")
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")

with open('pdf_data.json', 'w', encoding='utf-8') as json_file:
    json.dump(pdf_data, json_file, ensure_ascii=False)