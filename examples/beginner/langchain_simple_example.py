"""Research Assistant Agent (LangChain)

A Bindu agent that finds and summarizes information using LangChain.
Uses DuckDuckGo for web search capabilities.

Features:
- Web search integration via LangChain tools
- Information summarization
- Research assistance

Usage:
    python langchain_simple_example.py

Environment:
    Requires OPENAI_API_KEY in .env file
"""

import os

from bindu.penguin.bindufy import bindufy
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

# Define your LangChain agent
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
)

tools = [DuckDuckGoSearchRun()]

prompt = PromptTemplate.from_template(
    "You are a research assistant that finds and summarizes information.\n\n"
    "You have access to the following tools:\n{tools}\n\n"
    "Use the following format:\n"
    "Question: the input question you must answer\n"
    "Thought: you should always think about what to do\n"
    "Action: the action to take, should be one of [{tool_names}]\n"
    "Action Input: the input to the action\n"
    "Observation: the result of the action\n"
    "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
    "Thought: I now know the final answer\n"
    "Final Answer: the final answer to the original input question\n\n"
    "Begin!\n\n"
    "Question: {input}\n"
    "Thought:{agent_scratchpad}"
)

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Configuration
# Note: Infrastructure configs (storage, scheduler, sentry, API keys) are now
# automatically loaded from environment variables. See .env.example for details.
config = {
    "author": "your.email@example.com",
    "name": "langchain_research_agent",
    "description": "A LangChain-powered research assistant agent",
    "deployment": {
        "url": "http://localhost:3773",
        "expose": True,
        "cors_origins": ["http://localhost:5173"],
    },
    "skills": ["skills/question-answering"],
}


# Handler function
def handler(messages: list[dict[str, str]]):
    """Process messages and return agent response.

    Args:
        messages: List of message dictionaries containing conversation history

    Returns:
        Agent response result
    """
    user_input = messages[-1]["content"]
    result = agent_executor.invoke({"input": user_input})
    return result["output"]


# Bindu-fy it
if __name__ == "__main__":
    # Disable auth for local development - frontend can connect without OAuth
    os.environ["AUTH_ENABLED"] = "false"
    bindufy(config, handler)

# if you want to use tunnel to expose your agent to the internet, use the following command
# bindufy(config, handler, launch=True)
