import os
from dotenv import load_dotenv
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
# from langchain_core.output_parsers import StrOutputParser
import textwrap




load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)
# llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4-turbo", max_tokens=1024, temperature=0.1)
llm = ChatOpenAI(api_key=openai_api_key, max_tokens=1024, temperature=0.1)



def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def process_llm_response(llm_response):
    print(wrap_text_preserve_newlines(llm_response['result']))
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['Document'])

def main():
    query1 = "האם חברת אשראי יכולה לתת אשראי במטבע חוץ"
    # query1 = "מה הדרכים של חברות אשראי לגבות חוב?"
    # vectorstore = Pinecone_vectorstores.from_existing_index(embedding=embeddings_model, index_name='test-index-1536')
    vectorstore = Pinecone.from_existing_index(embedding=embeddings_model, index_name='test-index-1536')
    retriever =  vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm,retriever=retriever, return_source_documents=True, verbose=True)
    answer = qa_chain.invoke(query1)
    # print(StrOutputParser(answer['result']))
    # print(answer)
    # print(answer['source_documents'][1])
    process_llm_response(answer)


if __name__ == "__main__":
    main()