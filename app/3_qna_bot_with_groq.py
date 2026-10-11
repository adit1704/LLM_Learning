from dotenv import load_dotenv
from langgraph.checkpoint import memory
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st

llm=ChatGroq(model="openai/gpt-oss-20b",streaming=True)
search=GoogleSerperAPIWrapper()
tools=[search.run]

if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
    st.session_state.history = []

agent = create_agent(
    model=llm,
    tools=tools,
    checkpointer=st.session_state.memory,
    system_prompt="You are a helpful assistant that answers questions based on the information provided by the user. If you don't know the answer, just say you don't know. Do not try to make up an answer.",
)

#### Building WWEb interface using Streamlit
st.subheader("QnA Bot with Groq")

for message in st.session_state.history:
    if message["role"] == "user":
        st.chat_message("user").markdown(message["content"])
    else:
        st.chat_message("assistant").markdown(message["content"])

query = st.chat_input("Ask a question")
if query:
    st.chat_message("user").markdown(query)
    st.session_state.history.append({"role": "user", "content": query})
    response = agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        {"configurable": {"thread_id": "1"}},
        stream_mode="messages"
    )
    ai_container = st.chat_message("assistant")
    with ai_container:
        space = st.empty()
        answer = ""
        for chunk in response:
            answer += chunk[0].content # type: ignore
            space.write(answer)
        
    
    st.session_state.history.append({"role": "assistant", "content": answer})
