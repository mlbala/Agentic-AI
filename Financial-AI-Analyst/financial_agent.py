import os
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for information",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[DuckDuckGo()],
    instructions="Always include sources and references in your answers.",
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

## Define mulli model agent

multi_model_ai_agent = Agent(
    team=[web_search_agent, finance_agent],
    instructions=["Always include sources","Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

multi_model_ai_agent.print_response("Summarize analyst recommendations and share the latest news for NVDA", stream=True)