from concurrent.futures import thread

from dotenv import load_dotenv
load_dotenv()

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

llm = ChatGroq(model="openai/gpt-oss-20b")
search = GoogleSerperAPIWrapper()
memory = MemorySaver()

agent = create_agent(
    model = llm,
    tools = [search.run],
    system_prompt = "You are a helpful assistant that can answer questions using the Google search API.",
    checkpointer = memory
)



while True:
    user_input = input("User: ")
    if user_input.lower() == 'exit':
        break

    search_results = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        {"configurable": {"thread_id": "1"}}
        )
    print("AI: ", search_results['messages'][-1].content)