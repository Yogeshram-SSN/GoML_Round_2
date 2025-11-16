import json
import re
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load API Key
load_dotenv("api_keys.env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# Mock Flight Database
FLIGHT_DB = {
    "AI123": {"flight_number": "AI123", "departure_time": "08:00 AM", "destination": "Delhi", "status": "Delayed"},
    "BA204": {"flight_number": "BA204", "departure_time": "06:20 AM", "destination": "London", "status": "On Time"},
    "EK501": {"flight_number": "EK501", "departure_time": "03:45 PM", "destination": "Dubai", "status": "On Time"},
    "SQ318": {"flight_number": "SQ318", "departure_time": "10:15 AM", "destination": "Singapore", "status": "Boarding"},
    "DL45":  {"flight_number": "DL45",  "departure_time": "01:30 PM", "destination": "New York", "status": "Cancelled"},
    "AA101": {"flight_number": "AA101", "departure_time": "09:10 PM", "destination": "Chicago", "status": "On Time"},
    "QR571": {"flight_number": "QR571", "departure_time": "11:50 AM", "destination": "Doha", "status": "Delayed"},
    "LH760": {"flight_number": "LH760", "departure_time": "02:10 PM", "destination": "Frankfurt", "status": "On Time"},
    "CX701": {"flight_number": "CX701", "departure_time": "07:45 AM", "destination": "Hong Kong", "status": "Boarding"},
    "UK810": {"flight_number": "UK810", "departure_time": "05:25 PM", "destination": "Bangalore", "status": "On Time"},
}

# Info Agent
def get_flight_info(flight_number: str) -> dict:
    return FLIGHT_DB.get(flight_number)


def info_agent_request(flight_number: str) -> str:
    info = get_flight_info(flight_number)
    if info is None:
        return json.dumps({"error": "Flight not found"})
    return json.dumps(info)

# Regex Pattern Matching
FLIGHT_PATTERN = r"\b([A-Z]{2,3})\s*([0-9]{2,4})\b"

def clean_flight_number(raw: str) -> str:
    if not raw:
        return None
    cleaned = re.sub(r"[^A-Z0-9]", "", raw.upper())   # Remove symbols/spaces
    return cleaned if cleaned else None

def fallback_extract_with_regex(query: str):
    print("fallback used")
    match = re.search(FLIGHT_PATTERN, query, re.IGNORECASE)
    if match:
        airline = match.group(1)
        number = match.group(2)
        return clean_flight_number(airline + number)
    return None

# LLM Flight Number Extraction using OpenAI
def extract_flight_number_llm(user_query: str) -> str:
    prompt = f"""
Extract ONLY the flight number from the user query.

Valid global flight number format:
- 2 to 3 letters + 2 to 4 digits (spaces allowed)

Examples:
AI123, AI 123, BA204, EK501, SQ318, LH760, CX701, UK810

User query: "{user_query}"

Respond ONLY in JSON:
{{
  "flight_number": "<value or null>"
}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        text = response.choices[0].message.content.strip()
        data = json.loads(text)
        flight_no = data.get("flight_number")

        flight_no = clean_flight_number(flight_no)

        if not flight_no:
            return fallback_extract_with_regex(user_query)

        return flight_no

    except:
        return fallback_extract_with_regex(user_query)

# QA Agent
def qa_agent_respond(user_query: str) -> str:
    flight_no = extract_flight_number_llm(user_query)

    if not flight_no:
        return json.dumps({"answer": "No valid flight number found in query."})

    info_json = info_agent_request(flight_no)
    info = json.loads(info_json)

    if "error" in info:
        return json.dumps({"answer": f"Flight {flight_no} not found in database."})

    answer_text = (
        f"Flight {info['flight_number']} departs at {info['departure_time']} "
        f"to {info['destination']}. Current status: {info['status']}."
    )

    return json.dumps({"answer": answer_text})

# Main module
if __name__ == "__main__":
    print("Airline Assistant: Ask me about ANY flight!")
    print("Type 'exit' to quit.\n")

    while True:
        user_query = input("Enter your query: ")

        if user_query.lower() in ["exit", "quit", "bye"]:
            print("Bye!")
            break

        response = qa_agent_respond(user_query)
        print("\nResponse:", response, "\n")