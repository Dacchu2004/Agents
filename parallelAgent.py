import os
import asyncio
from dotenv import load_dotenv
from google.adk.agents import Agent, SequentialAgent, ParallelAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types

# ENVIRONMENT SETUP
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file")
os.environ["GOOGLE_API_KEY"] = api_key

retry_config = types.HttpRetryOptions(
    attempts=5, exp_base=7, initial_delay=1, http_status_codes=[429, 500, 503, 504]
)

# RESEARCH AGENTS (RUN IN PARALLEL)
tech_agent = Agent(
    name="TechResearcher",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""Research 3 key AI/ML trends with companies and impact (100 words).""",
    tools=[google_search],
    output_key="tech_research",
)

health_agent = Agent(
    name="HealthResearcher",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""Research 3 major medical breakthroughs and applications (100 words).""",
    tools=[google_search],
    output_key="health_research",
)

finance_agent = Agent(
    name="FinanceResearcher",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""Research 3 major fintech trends with implications (100 words).""",
    tools=[google_search],
    output_key="finance_research",
)

# Aggregator (runs after parallel agents)
aggregator = Agent(
    name="AggregatorAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""Combine research findings:
    - Tech: {tech_research}
    - Health: {health_research}
    - Finance: {finance_research}
    Write a 200-word executive summary highlighting key themes and takeaways.""",
    output_key="executive_summary",
)

# Parallel and Sequential workflow
parallel_team = ParallelAgent(
    name="ParallelResearchTeam",
    sub_agents=[tech_agent, health_agent, finance_agent],
)

root_agent = SequentialAgent(
    name="ResearchSystem",
    sub_agents=[parallel_team, aggregator],
)

# RUNNER
async def main():
    runner = InMemoryRunner(agent=root_agent)
    await runner.run_debug("Run the daily executive briefing on Tech, Health, and Finance.")

if __name__ == "__main__":
    asyncio.run(main())
