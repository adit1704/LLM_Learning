from dotenv import load_dotenv
load_dotenv()


## import required Liberaries
from pydantic import BaseModel, Field
from typing import Literal
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper

## define the class
class FlowState(BaseModel):
    question : str = Field(description="User asked Question")
    category : Literal["coding", "google search", "weather"] = Field(default="google search")
    ans : str = Field(default="")

class QuestionCategory(BaseModel):
    category : Literal["coding", "google search", "weather"] = Field(default="google search", description="Question category")

llm = ChatGroq(model="openai/gpt-oss-20b")

## search agents
search = GoogleSerperAPIWrapper()
search_tools = [search.run]

search_agent = create_agent(
    model = llm,
    tools= search_tools,
    system_prompt="You are a google agent, you can search for anything in google"
)

## Weather Agent
@tool
def get_weather(city:str):
    """weather details"""
    return f"the current temp in {city} is 23 degrees celsius"

weather_agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="You are a agent to tell weather"
)

## define functions / nodes
def check_question_category(state:FlowState) -> FlowState:
    st_llm = llm.with_structured_output(QuestionCategory)
    res = st_llm.invoke(f"Categorize my question {state.question} if not sure, just say google searchas category")
    state.category = res.category
    return state

def route(state: FlowState) -> Literal["coding", "google search", "weather"]:
    return state.category

def coding_node(state:FlowState) -> FlowState:
    res = llm.invoke(f"You are a coding expert. {state.question}")
    state.ans = str(res.content)
    return state

def Google_search_node(state:FlowState) -> FlowState:
    res = search_agent.invoke(
        {"messages":[{"role":"user", "content": state.question}]}
    )
    state.ans = res["messages"][-1].content
    return state

def weather_node(state:FlowState) -> FlowState:
    res = weather_agent.invoke({"messages":[
        {"role":"user", "content":state.question}
    ]})
    state.ans = res["messages"][-1].content
    return state

graph = StateGraph(FlowState)

graph.add_node("check question category", check_question_category)
graph.add_node("coding", coding_node)
graph.add_node("google search", Google_search_node)
graph.add_node("weather", weather_node)

graph.add_edge(START, "check question category")
graph.add_conditional_edges("check question category", route)
graph.add_edge("coding", END)
graph.add_edge("google search", END)
graph.add_edge("weather", END)

compiled_graph = graph.compile()


response = compiled_graph.invoke(FlowState(question="What is the temperature in Delhi"))
print(response)







