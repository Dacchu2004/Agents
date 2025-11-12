import os
import asyncio
from dotenv import load_dotenv
from google.adk.agents import Agent, SequentialAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
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

# STORY AGENTS

# Helper function to exit the loop
def exit_loop():
    return {"status": "approved", "message": "Story approved. Exiting loop."}

# Writer Agent (initial draft)
initial_writer = Agent(
    name="InitialWriterAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""Write the first draft of a 150-word short story based on the prompt.
    Output only the story text, no intro or explanation.""",
    output_key="current_story",
)

# Critic Agent
critic_agent = Agent(
    name="CriticAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""You are a story critic. Review this story:
    {current_story}
    If it's excellent, reply EXACTLY 'APPROVED'.
    Otherwise, give 2–3 actionable improvement suggestions.""",
    output_key="critique",
)

# Refiner Agent
refiner_agent = Agent(
    name="RefinerAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""Review the critique:
    {critique}
    If the critique is 'APPROVED', call exit_loop.
    Otherwise, rewrite the story to improve it.""",
    tools=[FunctionTool(exit_loop)],
    output_key="current_story",
)

# Loop agent for iterative refinement
story_refinement_loop = LoopAgent(
    name="StoryRefinementLoop",
    sub_agents=[critic_agent, refiner_agent],
    max_iterations=2,
)

# Root pipeline
story_pipeline = SequentialAgent(
    name="StoryPipeline",
    sub_agents=[initial_writer, story_refinement_loop],
)

# RUNNER
async def main():
    runner = InMemoryRunner(agent=story_pipeline)
    await runner.run_debug(
        "Write a short story about a lighthouse keeper who discovers a mysterious glowing map."
    )

if __name__ == "__main__":
    asyncio.run(main())
