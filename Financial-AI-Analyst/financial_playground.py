
from phi.agent import Agent
from phi.model.openai import OpenAIChat 
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from phi.playground import Playground, serve_playground_app
from phi.model.groq import Groq

import phi
import  phi.api
import openai

import os
from dotenv import load_dotenv


# Load the environment variables from the .env file
load_dotenv()

phi.api=os.getenv("PHI_API_KEY")


openai_api_key = os.getenv("OPENAI_API_KEY")

# web_search_agent = Agent(
#     name="Web Search Agent",
#     role="Search the web for information",
#     model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
#     tools=[DuckDuckGo()],
#     instructions="Always include sources and references in your answers.",
#     show_tool_calls=True,
#     markdown=True,
# )

web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for information",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources and references in your answers."],
    show_tool_calls=True,
    markdown=True,
)


## Financial Agent

finance_agent = Agent(
    name="Financial AI Agent",
    role="Financial Analyst",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)


## Playgorund

app=Playground(agents=[finance_agent, web_search_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("financial_playground:app", reload=True)
