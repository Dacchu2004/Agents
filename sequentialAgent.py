import os
import asyncio
from dotenv import load_dotenv
from google.adk.agents import Agent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
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

# AGENTS

# Outline Agent
outline_agent = Agent(
    name="OutlineAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""Create a blog outline for the given topic including:
    - A catchy headline
    - An engaging intro
    - 3–5 main sections (with 2–3 bullet points each)
    - A short conclusion.""",
    output_key="blog_outline",
)

# Writer Agent
writer_agent = Agent(
    name="WriterAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""Using this outline: {blog_outline}
    Write a 200–300 word blog post that is informative and engaging.""",
    output_key="blog_draft",
)

# Editor Agent
editor_agent = Agent(
    name="EditorAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""Polish this blog draft for grammar, tone, and flow:
    {blog_draft}""",
    output_key="final_blog",
)

# Sequential Pipeline
blog_pipeline = SequentialAgent(
    name="BlogPipeline",
    sub_agents=[outline_agent, writer_agent, editor_agent],
)

# RUNNER
async def main():
    runner = InMemoryRunner(agent=blog_pipeline)
    await runner.run_debug(
        "Write a blog post about the benefits of multi-agent systems for software developers."
    )

if __name__ == "__main__":
    asyncio.run(main())
