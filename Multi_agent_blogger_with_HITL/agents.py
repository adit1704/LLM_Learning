from dotenv import load_dotenv
load_dotenv()

from state import BlogState
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate

## get LLM
def get_llm(model_name : str = "openai/gpt-oss-20b", temperature : float = 0.5):
    return ChatGroq(model=model_name, temperature=temperature)

##RESEARCHHER AGENT
RESEARCH_PROMPT = ChatPromptTemplate.from_messages([
    {"role": "system", "content" :"""
    You are a researcher agent. Given a blog topic and target audience, produce a clear, structured research outline. Include
    1. 5-7 key points the blog should cover
    2.  Important facts, stats or examples  for each point 
    3.  Suggested angle  or hook
    Be concise, Use bullet points . Do NOT write a full blog yet
    """},
    {"role" : "user", "content" : "Topic : {topic}, Audience : {audience}, {revision_hints}, write the research outline now"}

])

def research_agent(llm: ChatGroq, topic: str, audience: str, feedback: str):
    revision_hints=f"the human provided this feedback on your previous research - please address it: {feedback}"
    if not feedback:
        revision_hints = "This is your first attempt"

    chain = RESEARCH_PROMPT | llm

    result = chain.invoke({
        "topic": topic,
        "audience": audience,
        "revision_hints": revision_hints
    })

    return result.content

##WRITER AGENT
WRITER_PROMPT = ChatPromptTemplate.from_messages([
    {"role": "system", "content" :"""
    You are a Blog Writer agent. Using the research notes provided, write an engaging blog
    Rules:
    1. Length 500-700 words
    2. Structure, catchy title, intro hook, 3-5 sections with H2 headings, conclusion
    3. Tone: clear friendly, suited for target audience
    4. Use Markdown formatting
    5. Do NOT add a word count at end
    """},
    {"role" : "user", "content" : "Topic : {topic}, Audience : {audience}, Research Notes {research} {revision_hints}, write the blog now"}

])

def writer_agent(llm: ChatGroq, topic: str, audience: str, research: str, feedback: str):
    revision_hints=f"the human provided this feedback on your previous blog - please address it: {feedback}"
    if not feedback:
        revision_hints = "This is your first attempt"

    chain = WRITER_PROMPT | llm

    result = chain.invoke({
        "topic": topic,
        "audience": audience,
        "research": research,
        "revision_hints": revision_hints
    })

    return result.content

##EDITOR AGENT
EDITOR_PROMPT = ChatPromptTemplate.from_messages([
    {"role": "system", "content" :"""
    You are an Editor agent - the final quality gate before publishing.
    Take the draft and produce the FINAL polished version, Specifically
    - fix the grammer, spelling and awkward phrasing
    - tighten wordy sentences
    - improve  flow  and transition between sections 
    - make the title and intro more complelling if needed
    - keep the same structure and markdown formatting
    Output only the final polished blog post. No commentary    
    """},
    {"role" : "user", "content" : "Topic : {topic},  draft {draft}  edit the blog now"}

])

def editor_agent(llm: ChatGroq, topic: str,  draft: str):
    
    chain = EDITOR_PROMPT | llm

    result = chain.invoke({
        "topic": topic,
        "draft": draft
    })

    return result.content