
### IMPORT DEPENDENCIES ###

# import basics
import os
from dotenv import load_dotenv

# import streamlit
import streamlit as st

# import langchain
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain.agents import create_tool_calling_agent
from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool

# import supabase db
from supabase.client import Client, create_client

### SET UP BACKEND FUNCITONALITY ###

# load environment variables
load_dotenv()  

# initiating supabase
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# initiating embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# initiating vector store
vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
)

# Get the directory where the Streamlit app is running
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the persistent storage folder inside the app directory
UPLOAD_DIR = os.path.join(APP_DIR, "documents")

# Ensure the directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Define persistent storage directory for URLs
URL_DIR = os.path.join(APP_DIR, "URLs")

# Ensure the directory exists
os.makedirs(URL_DIR, exist_ok=True)

# Store URLs in a persistent file
url_file_path = os.path.join(URL_DIR, "articles.txt")

### SET UP STREAMLIT APP ###

# initiating streamlit app
st.set_page_config(page_title="Agentic RAG Chatbot", 
                   page_icon="🤖",
                   layout="wide")

# create side bar
with st.sidebar:
        st.title('LLM:')
        
        modelo = st.radio(
            "Choose the LLM:",
            ["gpt-3.5", "gpt-4", "gpt-4o", "gpt-4o-mini"],
            index=2)
        
        if modelo == "gpt-3.5":
            llm = 'gpt-3.5-turbo-0125'
        elif modelo == "gpt-4":
            llm = 'gpt-4-turbo'
        elif modelo == 'gpt-4o':
            llm = 'gpt-4o'
        else:
            llm = 'gpt-4o-mini'


        st.divider()
        
        # Display available documents in the local file repository
        st.markdown("### Documents in the Vector Store:")
        
        # List all files in the /documents folder
        all_files = os.listdir(UPLOAD_DIR)

        # For document checkboxes
        selected_files = [file for idx, file in enumerate(all_files) if st.checkbox(file, value=True, key=f"doc_{idx}")]

        st.divider()            

        # Display available articles in the vector store
        st.markdown("### Articles in the Vector Store:")
        
        # Read and display articles from the stored file
        if os.path.exists(url_file_path):
            with open(url_file_path, "r") as f:
                all_articles = f.readlines()
        
                # For article checkboxes
                selected_articles = [article.strip() for idx, article in enumerate(all_articles) if st.checkbox(article.strip(), value=True, key=f"article_{idx}")]

# layout for main page
col1, col2 = st.columns([1, 5])

with col1:
   st.image('inline-logo-with-tagline-and-path-2048x1152.png')

with col2:
   st.header('Agentic RAG Chatbot')

 
# initiating llm
llm = ChatOpenAI(model=llm,temperature=0)

# pulling prompt from hub
prompt = hub.pull("hwchase17/openai-functions-agent")


# creating the retriever tool
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve from vector store information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=5)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    print(serialized)
    return serialized, retrieved_docs

# combining all tools
tools = [retrieve]

# initiating the agent
agent = create_tool_calling_agent(llm, tools, prompt)

# create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat messages from history on app rerun
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)


# create the bar where we can type messages
user_question = st.chat_input("Please write your question here")


# did the user submit a prompt?
if user_question:

    # add the message from the user (prompt) to the screen with streamlit
    with st.chat_message("user"):
        st.markdown(user_question)

        st.session_state.messages.append(HumanMessage(user_question))


    # invoking the agent
    result = agent_executor.invoke({"input": user_question, "chat_history":st.session_state.messages})

    print(result)
    
    ai_message = result["output"]

    # adding the response from the llm to the screen (and chat)
    with st.chat_message("assistant"):
        st.markdown(ai_message)

        st.session_state.messages.append(AIMessage(ai_message))

