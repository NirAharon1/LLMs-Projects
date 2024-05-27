from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
import sys
import pinecone
import glob
import PyPDF2 
import json
from tqdm import tqdm

embeddings_model_name="text-embedding-3-small" # US$0.02 / 1M tokens
# embeddings_model_name="text-embedding-ada-002" # US$0.10 / 1M tokens

script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
embeddings_model = OpenAIEmbeddings(model=embeddings_model_name, openai_api_key=openai_api_key)
pdf_folder = "PDF_folder"


pc = Pinecone(api_key=pinecone_api_key)



index_name = 'test-index-1536'
if index_name not in pc.list_indexes().names():
    pc.create_index(
            index_name, 
            dimension=1536, 
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            ))
index = pc.Index(index_name)


def get_pdf_text(pdf_path) -> str:
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text()
            text = text[0:230000] #limiting the text size
    return text


def get_text_chunks(text) -> list[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_embedding(text_chanks) ->list[list[float]]:
    openai_embedding = embeddings_model.embed_documents(text_chanks)
    return openai_embedding


with open('pdf_data.json', 'r', encoding='utf-8') as json_file:
    json_content = json_file.read()
    pdf_data = json.loads(json_content)
    try:
        for pdf_name in tqdm(pdf_data):
            pdf_path = os.path.join(pdf_folder, pdf_name)
            if not os.path.isfile(pdf_path):
                print(f"File not found: {pdf_path}")
                continue
            pdf_text = get_pdf_text(pdf_path)
            pdf_text_chunked = get_text_chunks(pdf_text)
            pdf_text_chunked_embedded = get_vector_embedding(pdf_text_chunked)
            chunk_ind_list =  [pdf_name+str(i) for i in range(len(pdf_text_chunked_embedded))]
            metadata = [{'text': paragraph} for paragraph in pdf_text_chunked]
            for dictionary in metadata:
                dictionary.update(pdf_data[pdf_name])
            index.upsert(vectors=zip(chunk_ind_list,pdf_text_chunked_embedded,metadata))
    except Exception as e:
        error_code = getattr(e, 'code', None)  # Get the error code if available
        print(f"An error occurred during zipping. Error code: {error_code}. Error message: {e}")


print('Total vector count upsert vector databae:' ,index.describe_index_stats()['total_vector_count'])
