import os
from dotenv import load_dotenv
import asyncio
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types

#Loading .env file
load_dotenv()

#Accessing Gemini API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file")

#retrying options
retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)

#AI agents
#Prompt -> Agent -> Thought -> Action -> Observation -> Final Answer
#Defining our agent
root_agent = Agent(
    name="GenSearch",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    description="A simple agent that can answer general questions.",
    instruction="You are a helpful assistant. Use Google Search for current info or if unsure.",
    tools=[google_search],
)

#runner --> sends our messages to the agent, and handles its responses.
runner = InMemoryRunner(agent=root_agent)

#to run our agent
async def main():
    response = await runner.run_debug(
        "What is the Agent Development Kit from Google? What languages is it available in?"
    )

if __name__ == "__main__":
    asyncio.run(main())