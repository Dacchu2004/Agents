import os
import asyncio
from dotenv import load_dotenv

from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search, AgentTool
from google.adk.code_executors import BuiltInCodeExecutor



#Load Environment

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file")

print("🔑 Gemini API Key Loaded Successfully!")


#  2. Retry Configuration
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504]
)


#Custom Function Tools

def get_fee_for_payment_method(method: str) -> dict:
    """Looks up the transaction fee percentage for a given payment method.
    Returns {"status": "success", "fee_percentage": value} or {"status": "error", ...}
    """
    fee_database = {
        "platinum credit card": 0.02,
        "gold debit card": 0.035,
        "bank transfer": 0.01,
    }
    fee = fee_database.get(method.lower())
    return {"status": "success", "fee_percentage": fee} if fee is not None else {
        "status": "error",
        "error_message": f"Payment method '{method}' not found",
    }


def get_exchange_rate(base_currency: str, target_currency: str) -> dict:
    """Looks up and returns the exchange rate between two currencies."""
    rate_database = {
        "usd": {"eur": 0.93, "jpy": 157.50, "inr": 83.58}
    }
    base = base_currency.lower()
    target = target_currency.lower()
    rate = rate_database.get(base, {}).get(target)
    return {"status": "success", "rate": rate} if rate is not None else {
        "status": "error",
        "error_message": f"Unsupported currency pair: {base_currency}/{target_currency}",
    }


#Create Calculation Agent
calculation_agent = LlmAgent(
    name="CalculationAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
        You are a specialized calculator that ONLY responds with Python code.
        RULES:
        1. Respond ONLY with a Python code block.
        2. The code MUST calculate the result and print it.
        3. Do NOT explain anything. No text before/after code.
    """,
    code_executor=BuiltInCodeExecutor(),
)


#Create Enhanced Currency Agent
enhanced_currency_agent = LlmAgent(
    name="enhanced_currency_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
        You are a smart currency conversion assistant.

        Steps:
        1. Use get_fee_for_payment_method() to determine the fee.
        2. Use get_exchange_rate() to fetch conversion rate.
        3. If any tool returns error, report that clearly.
        4. DO NOT calculate manually. Use CalculationAgent to generate Python code for the computation.
        5. Final response must:
           - Mention converted amount.
           - Include fee percentage, fee amount, amount after fee, and exchange rate.
    """,
    tools=[
        get_fee_for_payment_method,
        get_exchange_rate,
        AgentTool(agent=calculation_agent),
    ],
)


#Run Agent Example
async def run_example():
    runner = InMemoryRunner(agent=enhanced_currency_agent)

    response = await runner.run_debug(
        "Convert 1,250 USD to INR using a Bank Transfer. Show me the precise calculation."
    )

    print("\n🧠 Final Response:")
    print(response)


#Main Runner
if __name__ == "__main__":
    asyncio.run(run_example())
