import os
import uvicorn

from fastapi import FastAPI
from langserve import add_routes

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda


# =========================================================
# GEMINI
# =========================================================

GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")
print("GEMINI_API_KEY loaded:", bool(GOOGLE_API_KEY))
print("GEMINI_API_KEY length:", len(GOOGLE_API_KEY) if GOOGLE_API_KEY else 0)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    api_key=GOOGLE_API_KEY,
    temperature=0
)


# =========================================================
# TOOLS
# =========================================================

@tool
def generate_meal_ideas(ingredients: str, preference: str = "any"):
    """
    Suggest 5 practical meal ideas using available ingredients.
    """

    prompt = f"""
You are MealCraft AI, a practical cooking assistant.

Available ingredients:
{ingredients}

Food preference:
{preference}

Suggest exactly 5 meal ideas that can realistically be made using
the available ingredients.

For each meal provide:
1. Meal name
2. Main ingredients used
3. Short description
4. Difficulty: Easy / Medium

Prefer simple, practical meals.
Do not suggest ingredients that are difficult to find.
"""

    response = llm.invoke(prompt)
    return response.text


@tool
@tool
def generate_recipe(meal_name: str, ingredients: str):
    """Generate a simple recipe for a selected meal using available ingredients."""
    
    prompt = f"""
You are MealCraft AI, a practical cooking assistant.

Create a simple beginner-friendly recipe.

Meal: {meal_name}

Available ingredients: {ingredients}

Provide:
1. Ingredients required
2. Preparation time
3. Step-by-step cooking instructions
4. Simple cooking tips

Use the available ingredients as much as possible.
If an essential ingredient is missing, clearly mention it.

Keep the response concise and practical.
"""

    response = llm.invoke(prompt)
    return response.text


@tool
def suggest_substitutions(ingredient: str, available_ingredients: str):
    """
    Suggest practical substitutes for a missing ingredient.
    """

    prompt = f"""
You are MealCraft AI, a practical cooking assistant.

Missing ingredient:
{ingredient}

Ingredients currently available:
{available_ingredients}

Suggest 3 practical substitutes for the missing ingredient.

For each substitute provide:
1. Substitute
2. Why it works
3. How to use it

Prefer commonly available ingredients.
If no good substitute exists, clearly say so.
"""

    response = llm.invoke(prompt)
    return response.text


# =========================================================
# AGENT
# =========================================================

tools = [
    generate_meal_ideas,
    generate_recipe,
    suggest_substitutions
]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are MealCraft AI, a practical cooking assistant. "
        "Use the available tools when the user asks for meal ideas, recipes, "
        "or ingredient substitutions. "
        "Keep responses simple and practical. "
        "Do not repeatedly call tools. "
        "For questions unrelated to cooking, meals, recipes, or ingredients, "
        "say exactly: "
        "'I am not authorized to answer questions outside of cooking and meals.'"
    )
)


# =========================================================
# LANGSERVE INPUT / OUTPUT
# =========================================================

class AgentInput(BaseModel):
    input: str = Field(description="Your message to MealCraft AI")


def format_for_agent(x) -> dict:
    user_input = x["input"] if isinstance(x, dict) else x.input
    return {"messages": [("user", user_input)]}


def extract_text_response(agent_output: dict) -> str:

    if not isinstance(agent_output, dict):
        return str(agent_output)

    messages = agent_output.get("messages")

    if messages is None:
        model_output = agent_output.get("model")

        if isinstance(model_output, dict):
            messages = model_output.get("messages")

    if messages:
        last = messages[-1]

        content = getattr(last, "content", "")

        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    return block.get("text", "")

        return str(content)

    return str(agent_output)


formatted_agent_chain = (
    RunnableLambda(format_for_agent)
    | agent
    | RunnableLambda(extract_text_response)
).with_types(
    input_type=AgentInput,
    output_type=str
)


# =========================================================
# FASTAPI + LANGSERVE
# =========================================================

app = FastAPI(
    title="MealCraft AI",
    version="1.0",
    description=(
        "A LangChain agent for meal ideas, recipes, "
        "and ingredient substitutions."
    ),
)


@app.get("/")
def root():
    return {
        "message": (
            "MealCraft AI server is running. "
            "Visit /agent/playground/ to chat."
        )
    }


add_routes(
    app,
    formatted_agent_chain,
    path="/agent"
)


# =========================================================
# SERVER
# =========================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
