# agent/agent_core.py (Version 4 - Final)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_react_agent, AgentExecutor

from .prompts import prompt

def create_agent(tavily_api_key: str, model_name: str = "gemini-1.5-flash"):
    """
    Initializes and returns a ReAct agent with a search tool.
    """
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0
    )

    tools = [
        TavilySearchResults(max_results=5, api_key=tavily_api_key)
    ]

    agent = create_react_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True,
        verbose=True,
        return_intermediate_steps=True
    )

    return agent_executor
