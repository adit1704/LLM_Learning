from dotenv import load_dotenv
load_dotenv()

### db, llm, tools, agent, memory, system_prompt, and streamlit imports
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st

db = SQLDatabase.from_uri("sqlite:///database.db")
db.run("""
    CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        description TEXT,
        status TEXT CHECK (status IN ('pending', 'in_progress', 'completed')) DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
""")

##llm, tools, agent, memory, system_prompt
llm = ChatGroq(model="openai/gpt-oss-20b")
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

system_prompt="""
You are a task management assistant. You can help users manage their tasks in a SQLite database. You can create, read, update, and 
delete tasks. Each task has a title, description, status (pending, in_progress, completed), and a created_at timestamp. 
Use the provided tools to interact with the database.
When responding to user queries, provide clear and concise answers. If you don't know the answer
"""

@st.cache_resource
def create_Agent():
    agent = create_agent(
        model=llm,
        tools=tools,
        checkpointer=InMemorySaver(),
        system_prompt=system_prompt
    )
    return agent

agent = create_Agent()

st.subheader("Task Management Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []  

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

prompt = st.chat_input("Ask me to manage your tasks")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            response = agent.invoke(
                {"messages": [{"role": "user", "content": prompt}]},
                {"configurable": {"thread_id": "1"}}
            )
            result = response["messages"][-1].content
            st.session_state.messages.append({"role": "assistant", "content": result})
            st.markdown(result)
    