import os
import json

from dotenv import load_dotenv
from google import genai

try:
    import streamlit as st
except ImportError:
    st = None

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

# Fallback to Streamlit Secrets when deployed
if not api_key and st is not None:
    api_key = st.secrets["GEMINI_API_KEY"]

client = genai.Client(api_key=api_key)


def generate_response(question: str):
    """
    Generate a medical answer along with the top 5 factual claims
    and optimized PubMed search queries.
    """

    prompt = f"""
You are an expert medical AI assistant.

Answer the user's question accurately.

Then identify ONLY the 5 most important factual medical claims from your answer.

For each claim, generate a concise PubMed search query using only the essential medical keywords.

Return ONLY valid JSON.

Format:

{{
    "answer": "Complete medical answer",

    "claims": [
        {{
            "claim": "Claim 1",
            "query": "Optimized PubMed search query"
        }},
        {{
            "claim": "Claim 2",
            "query": "Optimized PubMed search query"
        }},
        {{
            "claim": "Claim 3",
            "query": "Optimized PubMed search query"
        }},
        {{
            "claim": "Claim 4",
            "query": "Optimized PubMed search query"
        }},
        {{
            "claim": "Claim 5",
            "query": "Optimized PubMed search query"
        }}
    ]
}}

Rules:
- Return EXACTLY 5 claims.
- Claims must be factual and independently verifiable.
- Search queries should contain only important medical keywords.
- Do not include explanations.
- Do not wrap the JSON in markdown.
- Output valid JSON only.

User Question:
{question}
"""

    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt
    )

    text = response.text.strip()

    # Remove markdown fences if Gemini adds them
    if text.startswith("```json"):
        text = text.replace("```json", "").replace("```", "").strip()
    elif text.startswith("```"):
        text = text.replace("```", "").strip()

    return json.loads(text)
