# Multi-Agent Airline Information System (AWS Bedrock)

This project implements a *Two-Agent AI System* for answering flight-related queries:

### QA Agent  
Receives user questions like:
- "When does EK501 depart?"
- "What is the status of BA204?"

Uses AWS Bedrock to extract the *flight number* from natural-language text.

### Info Agent  
Provides structured flight information from a mock flight database.

---

## Features

- Supports ALL airlines (AI, BA, EK, DL, LH, QR, SQ, CX, etc.)
- Flight number extraction powered by *OpenAI GPT 4o*
- Regex fallback for reliability
- Strict JSON output format
- Expandable mock flight database
- Modular and easy to use

---

## File Structure

├── main.py <br/>
├── api_keys.env <br/>
├── requirements.txt <br/>
├── README.md <br/>


---

## Setup

### Requirement: Python version >= 3.10

### 1. Create Virtual Environment
bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate


### 2. Install Dependencies
pip install -r requirements.txt

### 3. Add API Keys
Inside api_keys.env:

OPENAI_API_KEY=YOUR_KEY <br/>

### 4. Run the script
python main.py