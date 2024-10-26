"""
RAG LLM app with history-aware retriever
Based on vector searching of the Bank Of Israel's "Proper conduct of banking business directive"
"""

import os
from typing import List

import streamlit as st
from streamlit_float import float_init, float_css_helper, float_parent
from streamlit_feedback import streamlit_feedback
from langchain.schema import HumanMessage, AIMessage
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.callbacks import collect_runs
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.documents import Document
from langchain_core.runnables import chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langsmith import Client

from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
langsmith_client = Client(api_key=langchain_api_key)

MODEL_NAME = "gpt-4o-mini" # $0.150 / 1M input tokens
# MODEL_NAME = "gpt-4o" # $2.50 / 1M input tokens
# MODEL_NAME = "gpt-4-turbo" # $10.00 / 1M tokens
# MODEL_NAME = "gpt-3.5-turbo" # $3.000 / 1M input tokens

EMBEDDINGS_MODEL_NAME="text-embedding-3-small" # US$0.02 / 1M tokens
# EMBEDDINGS_MODEL_NAME="text-embedding-ada-002" # US$0.10 / 1M tokens

INDEX_NAME = 'test-index-1536'
MAX_TOKENS = 300

st.set_page_config(page_title="bot", page_icon="🤖")
float_init(theme=True, include_unstable_primary=False)
st.title("AI bot - מסמכי נוהל בנקאי תקין")

@st.cache_resource
def local_css(file_name: str) -> None:
    """Loads and applies custom CSS styles from a specified file."""
    try:
        with open(file_name, encoding='utf-8') as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileExistsError:
        st.error(f"CSS file {file_name} not found")
local_css("style.css") # Load custom CSS

session_defaults = {
    "number_of_docs":1,
    "similarity_score_threshold":0.65,
    "chat_history":[],
    "contents":[],
    "retrived_content":[],
    "run_id":None,
    "feedback_result" :None

}

for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

@st.cache_resource(show_spinner=False)
def initialize_llm() -> ChatOpenAI:
    """Initializes and returns a language model instance for use in the application."""
    return ChatOpenAI(
                api_key=openai_api_key,
                model=MODEL_NAME,
                max_tokens=MAX_TOKENS,
                temperature=0.1,
                streaming=True,  # Enable streaming
                )

@st.cache_resource(show_spinner=False)
def initialize_retriever(number_of_docs: int,  threshold:float):
    """
    Initializes and returns a document retriever configured for similarity searches.
    This function creates an OpenAI embeddings model and a Pinecone vector store.
    Find similar documents based on the specified number of documents to retrieve.
    """
    embeddings_model = OpenAIEmbeddings(model=EMBEDDINGS_MODEL_NAME, openai_api_key=openai_api_key)
    vectorstore = PineconeVectorStore.from_existing_index(embedding=embeddings_model, index_name=INDEX_NAME)
                                                          
    @chain
    def retriever_with_similarity_score(query: str, n_docs=number_of_docs, threshold_float=threshold) -> List[Document]:
        try:
            docs, scores = zip(*vectorstore.similarity_search_with_relevance_scores(query, k=n_docs,score_threshold=threshold_float))
            for doc, score in zip(docs, scores):
                doc.metadata["score"] = "{:.1%}".format(score)
        except ValueError:
            print("No ducoument retriverd due to threshold score limit")
            return []
        return docs
    return retriever_with_similarity_score
    # return vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': number_of_docs})

def initialize_combined_chain(llm, retriever):
    """
    Initializes and returns a combined retrieval and question-answering chain.
    It sets up the necessary prompts for processing user queries and retrieving relevant context.
    """
    history_template = """Given a chat history and the latest user question which might reference context in the chat history, 
                        formulate a standalone question which can be understood without the chat history. Do NOT answer the question, 
                        just reformulate it if needed and otherwise return it as is. 
                        Chat History: {chat_history}
                        """
    history_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", history_template),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}")
                    ]
                )
    
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
    history_chain = create_history_aware_retriever(llm, retriever, history_prompt)
    return create_retrieval_chain(history_chain, qa_chain)



def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if "chat_message_history" not in st.session_state:
        st.session_state.chat_message_history = ChatMessageHistory()
    return st.session_state.chat_message_history

def display_ai_response_context(context):
    expander_style = """<style>@import "styles.css";</style>"""
    for i, ai_response_content in enumerate(context):
        expander_title = f"**{i+1}. {ai_response_content.metadata['title']} ({ai_response_content.metadata['date']})**  -   **Similarity score: {ai_response_content.metadata['score']}**"
        st.markdown(expander_style, unsafe_allow_html=True)
        with st.expander(expander_title, expanded=False, icon=':material/picture_as_pdf:'):
            st.write(ai_response_content.page_content)
            st.write(ai_response_content.metadata['pdf_url'])

def display_chat_history():
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("human"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("ai"):
                st.markdown(message.content)
        else:
            display_ai_response_context(message)
     

def chat_content_append():
    st.session_state['contents'].append(st.session_state.content) 

def generate_response(chain, query):
    for chunk in chain.stream({"input": query, "chat_history": st.session_state.chat_history}, config={"configurable": {"session_id": INDEX_NAME}}):
        if (input_data := chunk.get("input")) is not None:
            pass
        if (context := chunk.get("context")) is not None:
            st.session_state.retrived_content = context
        if (answer_chunk := chunk.get("answer")) is not None:
            yield answer_chunk  # Yield each chunk of the answer


def processes_feedback(run_id=0):
    score_mappings = {"👍": 1, "👎": 0,}
    if st.session_state.feedback_result is not None:
        thumb_score = st.session_state.feedback_result["score"]
        score = score_mappings[thumb_score]
        feedback_response = langsmith_client.create_feedback(
                run_id,
                key="feedback-key",
                score=score
                )
        print(f"feeback create for llm response {run_id} with score {score}/1")


def feedback(id):
    with st.form(key='fb_form',border=False):
        streamlit_feedback(
            feedback_type="thumbs",
            key="feedback_result",
            # key=f"feedback_{run_id}",
            )
        st.form_submit_button('Send', on_click=processes_feedback, args=(id,))


def main()->None:
    llm = initialize_llm()
    retriever = initialize_retriever(st.session_state.number_of_docs, st.session_state.similarity_score_threshold)
    combined_chain = initialize_combined_chain(llm, retriever)
    final_chain = RunnableWithMessageHistory(
        combined_chain,
        get_session_history,
        input_messages_key="input", 
        output_messages_key="answer",
        history_messages_key="chat_history",
        )
    

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
                st.rerun()

 
            st.number_input(
                label="Similarity score threshold",
                min_value=0.01,
                max_value=1.00,
                key="similarity_score_threshold",
                step=0.05,
            )

    with col2:
        with st.container(border=False):
            with st.container():
                user_query = st.chat_input("מה שאלתך? ", key='content', on_submit=chat_content_append, max_chars=4000)
                button_css = float_css_helper(width="2.2rem", bottom="0rem", transition=0)
                float_parent(css=button_css)

            display_chat_history()

            if user_query is not None and user_query !="":
                st.empty()
                st.session_state.chat_history.append(HumanMessage(user_query))

                with st.chat_message("human"):
                    st.markdown(user_query)
                
                with st.chat_message("ai"):
                    with collect_runs() as cb:
                        ai_response = st.write_stream(generate_response(final_chain, user_query))
                        run_id = cb.traced_runs[0].id

                    
                        st.session_state.chat_history.append(AIMessage(ai_response))
                        st.session_state.chat_history.append(st.session_state.retrived_content)
                        display_ai_response_context(st.session_state.retrived_content)
            

                        if run_id is not None:
                            feedback(run_id)


                        





if __name__ == "__main__":
    main()

