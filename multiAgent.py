import os
import asyncio
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, google_search
from google.genai import types


# ENVIRONMENT SETUP

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file")
os.environ["GOOGLE_API_KEY"] = api_key

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# AGENT DEFINITIONS

# Research Agent
research_agent = Agent(
    name="ResearchAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""You are a research specialist. Use google_search to find 2–3
    relevant, cited pieces of information on the given topic.""",
    tools=[google_search],
    output_key="research_findings",
)

# Summarizer Agent
summarizer_agent = Agent(
    name="SummarizerAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""Summarize these research findings in 3–5 concise bullet points:
    {research_findings}""",
    output_key="final_summary",
)

# Root Coordinator Agent
root_agent = Agent(
    name="ResearchCoordinator",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""You are a research coordinator. Follow these steps:
    1. Call ResearchAgent to collect info.
    2. Call SummarizerAgent to summarize findings.
    3. Present a clear final summary.""",
    tools=[AgentTool(research_agent), AgentTool(summarizer_agent)],
)


# RUNNER

async def main():
    runner = InMemoryRunner(agent=root_agent)
    await runner.run_debug(
        "What are the latest advancements in quantum computing and how might they impact AI?"
    )

if __name__ == "__main__":
    asyncio.run(main())
