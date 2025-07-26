import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.tools import PythonAstREPLTool
from langchain.agents import create_react_agent, AgentExecutor
from dotenv import load_dotenv

from .prompts import prompt

# Load environment variables from the .env file
load_dotenv()

def create_agent(dataframe: pd.DataFrame):
    """
    Initializes and returns a ReAct agent with a Python REPL tool.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0
    )

    # 1. Create the Tool
    # The PythonAstREPLTool allows the agent to execute Python code.
    # We pass the dataframe `df` into the tool's local namespace.
    # This means the agent's code will have access to the variable `df`.
    python_tool = PythonAstREPLTool(locals={"df": dataframe})
    
    tools = [python_tool]

    # 2. Create the ReAct Agent
    # We are now using the standard create_react_agent function which is more stable.
    agent = create_react_agent(llm, tools, prompt)

    # 3. Create the Agent Executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True,
        verbose=True,
        return_intermediate_steps=True,
        # This is important to prevent the agent from getting stuck in loops
        max_iterations=5, 
    )

    return agent_executor
