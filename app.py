import os
# Avoid OpenMP runtime conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

# Compatibility fallback for create_agent across LangChain / LangGraph versions
try:
    from langchain.agents import create_agent
except (ImportError, ModuleNotFoundError):
    from langgraph.prebuilt import create_react_agent
    def create_agent(model, tools, system_prompt=None, checkpointer=None, **kwargs):
        return create_react_agent(
            model=model,
            tools=tools,
            prompt=system_prompt,
            checkpointer=checkpointer,
            **kwargs
        )


st.set_page_config(page_title="Agentic RAG with Mistral & Hugging Face", page_icon="🤖", layout="wide")
st.title("📄 Agentic RAG Assistant (Mistral AI + Hugging Face)")

@st.cache_resource
def get_embeddings():
    """Load and cache Hugging Face embedding model."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


### data in st session

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []


def process_document(path):
    ## load the documents
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()

    if not docs:
        st.error("No content could be extracted from the uploaded PDF(s).")
        return

    ## split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(documents=docs)

    ## Hugging Face embeddings and Vector DB
    embeddings = get_embeddings()
    vector_db = InMemoryVectorStore.from_documents(
        documents=split_docs,
        embedding=embeddings
    )
    st.session_state.vector_store = vector_db

    ## create Mistral LLM agent - tool, llm, prompt
    # Ensure MISTRAL_API_KEY is available
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_api_key:
        st.error("MISTRAL_API_KEY is not set. Please set it in your .env file or environment.")
        return

    llm = ChatMistralAI(model="mistral-large-latest", temperature=0)

    @tool
    def retrieve_context(query: str):
        """Retrieve documents relevant to a query from the knowledge base."""
        context = ""
        retrieved_docs = vector_db.similarity_search(query=query, k=3)
        for doc in retrieved_docs:
            context += doc.page_content + "\n\n"
        return context

    system_prompt = """You are a helpful assistant that answers questions using retrieved context. 
My knowledge base consists of the details from the uploaded document. 
ALWAYS use the `retrieve_context` tool for questions requiring external knowledge."""

    memory = InMemorySaver()

    agent = create_agent(
        model=llm,
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=memory
    )

    st.session_state.agent = agent
    st.session_state.document_uploaded = True


### upload ui
if not st.session_state.document_uploaded:
    st.subheader("Upload Documents")
    uploaded = st.file_uploader(label="Select PDF Files", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        with st.spinner("Processing documents with Hugging Face embeddings..."):
            path = "./doc_files/"
            os.makedirs(path, exist_ok=True)
            for file in uploaded:
                file_path = os.path.join(path, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getvalue())

            process_document(path)
            st.rerun()

            
## chat ui
if st.session_state.document_uploaded and st.session_state.agent:
    st.sidebar.success("✅ Documents indexed with Hugging Face!")
    st.sidebar.info("🤖 Agent powered by Mistral AI")
    if st.sidebar.button("Reset / Upload New Documents"):
        st.session_state.document_uploaded = False
        st.session_state.agent = None
        st.session_state.vector_store = None
        st.session_state.messages = []
        st.rerun()

    for message in st.session_state.messages:
        role = message.get("role")
        content = message.get("content")
        st.chat_message(role).markdown(content)

    query = st.chat_input("Ask anything related to uploaded documents....")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").markdown(query)

        with st.spinner("Mistral Agent Thinking..."):
            response = st.session_state.agent.invoke(
                {"messages": [{"role": "user", "content": query}]},
                {"configurable": {"thread_id": "1"}}
            )

            answer = response["messages"][-1].content
            st.chat_message("ai").markdown(answer)
            st.session_state.messages.append({"role": "ai", "content": answer})
