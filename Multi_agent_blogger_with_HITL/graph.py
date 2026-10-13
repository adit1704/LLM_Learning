from typing import Literal, cast

from langgraph.graph import StateGraph, START, END
from state import BlogState
from agents import research_agent, writer_agent, editor_agent, get_llm
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt


## Defining the Nodes

def researcher_node(state: BlogState) -> BlogState:
    llm = get_llm()

    research_data = research_agent(
        llm = llm,
        topic = state.topic,
        audience = state.audience,
        feedback = state.research_feedback
    )

    state.research = research_data
    return state

def human_review_after_research(state:BlogState):
    decision = interrupt({
        "stage":"research_review",
        "research":state.research,
        "instructions":("reply with'approve'  to continue writting or describe what changes to be done")
    })

    if isinstance(decision, dict):
        action = decision.get("action", "approve")
        feedback = decision.get("feedback", "")
    else:
        text = str(decision)
        action = "approve" if text.lower() in ["approve", "approved", "ok", "yes", "process", ""] else "revise"
        feedback = "" if action == "approve" else text                            

    state.research_feedback = feedback
    return state

def writer_node(state: BlogState) -> BlogState:
    llm = get_llm()

    draft = writer_agent(
        llm = llm,
        topic = state.topic,
        audience = state.audience,
        research = state.research,
        feedback = state.writer_feedback
    )
    state.writer_draft = draft
    return state

def human_review_after_draft(state:BlogState):
    decision = interrupt({
        "stage":"draft_review",
        "draft":state.writer_draft,
        "instructions":("reply with 'approve'  to continue editting or describe what changes to be done")
    })

    if isinstance(decision, dict):
        action = decision.get("action", "approve")
        feedback = decision.get("feedback", "")
    else:
        text = str(decision)
        action = "approve" if text.lower() in ["approve", "approved", "ok", "yes", "process", ""] else "revise"
        feedback = "" if action == "approve" else text          

    state.writer_feedback = feedback
    return state


def editor_node(state: BlogState) -> BlogState:
    llm = get_llm()

    final = editor_agent(
        llm = llm,
        topic = state.topic,
        draft = state.writer_draft
    )
    state.final_edit = final
    return state


## Add conditional Nodes
def route_after_research_review(state: BlogState) -> Literal ["research_agent", "writer_agent"]:
    if state.research_feedback:
        return "research_agent"
    return "writer_agent"

def route_after_writer_review(state: BlogState) -> Literal ["editor_agent", "writer_agent"]:
    if state.writer_feedback:
        return "writer_agent"
    return "editor_agent"


## Add Nodes
def build_graph():
    graph = StateGraph(BlogState)

    graph.add_node("research_agent", researcher_node)
    graph.add_node("research_review", human_review_after_research)
    graph.add_node("writer_agent", writer_node)
    graph.add_node("draft_review", human_review_after_draft)
    graph.add_node("editor_agent", editor_node)

    ## Add Edges

    graph.add_edge(START, "research_agent")
    graph.add_edge("research_agent", "research_review")
    graph.add_conditional_edges("research_review", route_after_research_review)
    graph.add_edge("writer_agent", "draft_review")
    graph.add_conditional_edges("draft_review", route_after_writer_review)
    graph.add_edge("editor_agent", END)

    return graph.compile(checkpointer=InMemorySaver())