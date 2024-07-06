import streamlit as st
from streamlit_float import *
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import ChatMessagePromptTemplate
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
import os

st.set_page_config(page_title="bot", page_icon="🤖")
float_init(theme=True, include_unstable_primary=False)


if "number_of_docs" not in st.session_state:
    st.session_state.number_of_docs = 3

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


embeddings_model_name="text-embedding-3-small" # US$0.02 / 1M tokens
# embeddings_model_name="text-embedding-ada-002" # US$0.10 / 1M tokens

model_mane ="gpt-4o"
# model_mane ="gpt-4-turbo"
# model_mane ="gpt-3.5-turbo"


index_name = 'test-index-1536'



load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
embeddings_model = OpenAIEmbeddings(model=embeddings_model_name, openai_api_key=openai_api_key)
# llm = ChatOpenAI(api_key=openai_api_key, model=model_mane, max_tokens=2048, temperature=0.1)
llm = ChatOpenAI(api_key=openai_api_key, model=model_mane, max_tokens=1024, temperature=0.1)



vectorstore = Pinecone.from_existing_index(embedding=embeddings_model, index_name=index_name)
retriever =  vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': st.session_state.number_of_docs})

history_template = """Given a chat history and the latest user question which might reference context in the chat history, 
                      formulate a standalone question which can be understood without the chat history. Do NOT answer the question, 
                      just reformulate it if needed and otherwise return it as is.
                             
                     Chat History: {chat_history}
                      """
history_prompt = ChatPromptTemplate.from_messages([("system", history_template), MessagesPlaceholder("chat_history"), ("human", "{input}")])
history_chain = create_history_aware_retriever(llm, retriever, history_prompt)


qa_template = """
            You are a helpful Legal advisory lawyer in the field of finance.
            Your job is to return complete and accurate answers to the manager of a consumer credit company.
            Answer the question based on the context provided below. 
            If you're unable to answer the question, simply reply with "Sorry, I don't know the answer". 
            When a user requests code output, respond with: 'Sorry, we don't allow code output.'
            Context: {context}
            Question: {input}  
                        
           """
qa_prompt = ChatPromptTemplate.from_messages([("system", qa_template), MessagesPlaceholder("chat_history"), ("human", "{input}")])
qa_chain = create_stuff_documents_chain(llm, qa_prompt)


# Use Streamlit session state for chat history management
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if "chat_message_history" not in st.session_state:
        st.session_state.chat_message_history = ChatMessageHistory()
    return st.session_state.chat_message_history

combined_chain = create_retrieval_chain(history_chain, qa_chain)
final_chain = RunnableWithMessageHistory(
    combined_chain,
    get_session_history,
    input_messages_key="input", 
    history_messages_key="chat_history",
    output_messages_key="answer"
    )




def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css("style.css") # Load custom CSS
st.title("AI bot - מסמכי נוהל בנקאי תקין")



def chat_content():
    st.session_state['contents'].append(st.session_state.content)

if 'contents' not in st.session_state:
    st.session_state['contents'] = []
    border = False
else:
    border = True



col1, col2 = st.columns([1, 4])
with col1:
    with st.container(border=False):
        st.number_input(
            label="Number of Documents",
            min_value=1,
            max_value=7,
            key="number_of_docs",
            step=1,
            format="%d"
        )

        if st.button("Restart Chat"):
            st.session_state.chat_history = []
            st.session_state.expander_states = {}
            st.experimental_rerun()


with col2:    
    with st.container(border=False): 
        with st.container():
            user_query = st.chat_input("מה שאלתך? ", key='content', on_submit=chat_content) 
            button_css = float_css_helper(width="2.2rem", bottom="0rem", transition=0)
            float_parent(css=button_css)
    
        #conversation
        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.markdown(message.content)
            elif isinstance(message, AIMessage):
                with st.chat_message("AI bot"):
                    st.markdown(message.content)
            elif isinstance(message, list):
                for i, ai_response_content in enumerate(message):
                    expander_style = """<style>@import "styles.css";</style>"""
                    st.markdown(expander_style, unsafe_allow_html=True)
                    with st.expander(f"**{i+1}. {ai_response_content.metadata['title']} ({ai_response_content.metadata['date']})**", expanded=False):
                        st.write(ai_response_content.page_content)
                        st.write(ai_response_content.metadata['pdf_url'])

        if user_query is not None and user_query !="":
            st.empty()
            st.session_state.chat_history.append(HumanMessage(user_query))

            with st.chat_message("Human"):
                st.markdown(user_query)
            
            with st.chat_message("AI"):
                with st.spinner("Generating..."):
                    ai_response = final_chain.invoke({"input": user_query, "chat_history": st.session_state.chat_history}, config={"configurable": {"session_id": index_name}})
                    ai_response_answer = ai_response["answer"]
                    st.markdown(ai_response_answer)
                    st.session_state.chat_history.append(AIMessage(ai_response_answer))
                    st.session_state.chat_history.append(ai_response["context"])


                for i, ai_response_content in enumerate(ai_response["context"]):
                    expander_style = """<style>@import "styles.css";</style>"""
                    expander_id = f"expander_{i}"
                    st.markdown(expander_style, unsafe_allow_html=True)
                    with st.expander(f"**{i+1}. {ai_response_content.metadata['title']} ({ai_response_content.metadata['date']})**", expanded=False):
                        st.write(ai_response_content.page_content)
                        st.write(ai_response_content.metadata['pdf_url'])









