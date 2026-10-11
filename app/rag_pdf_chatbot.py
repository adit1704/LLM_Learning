from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import InMemoryVectorStore
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
from langchain.agents import create_agent
import streamlit as st
import os

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "messages" not in st.session_state:
    st.session_state.messages = []

def process_document(path):

    loader = PyPDFDirectoryLoader(path)
    doc = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(doc)
    
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_db=InMemoryVectorStore.from_documents(
        documents=chunks,
        embedding=embedding
    )

    llm = ChatGroq(model="openai/gpt-oss-20b")

    @tool
    def get_context(query:str):
        """
        This tool takes a query as input and returns the context from the vector database based on the
        similarity search. It also returns the original query for further processing.
        """
        data = vector_db.similarity_search(query)
        context = ""
        for d in data:
            context += d.page_content + "\n"
        return context

    system_prompt = """
    You are a helpful assistant. Answer the question based on the context below. If the answer is not contained within the text below, say "I don't know".
    ALWAYS use the get_context tool to retrieve the context for the question before answering. Do not answer the question without using the tool.
    """
    memory = InMemorySaver()
    agent = create_agent(
        model=llm,
        tools=[get_context],
        system_prompt=system_prompt,
        checkpointer=memory
    )
    st.session_state.agent = agent
    st.session_state.document_uploaded = True

## Upload UI
if not st.session_state.document_uploaded:
    st.subheader("Upload a PDF document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)
    if uploaded_file:
        with st.spinner("Processing document..."):
            path = "./doc_files/"
            for file in uploaded_file:
                with open(path + file.name, "wb") as f:
                    f.write(file.getvalue())
            process_document(path)
            st.rerun()

# Chat UI
if st.session_state.document_uploaded and st.session_state.agent:
    st.subheader("Ask questions about the document")

    for message in st.session_state.messages:
        role = message.get("role")
        content = message.get("content")
        st.chat_message(role).markdown(content)

    query = st.chat_input("Ask a question")
    if query:
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})
        response = st.session_state.agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            {"configurable": {"thread_id": "1"}}
        )
        answer = response["messages"][-1].content
        st.chat_message("AI").markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

