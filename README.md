# ATLAS - Autonomous Transition & Longitudinal Agent System

A multi-agent healthcare AI system for care transitions, built with Groq API and FHIR R4.

## Version
1.0.0

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment:
Copy `.env` and set your `GROQ_API_KEY`.

## Running

Start the API:
```bash
uvicorn main:app --reload
```

## Testing

```bash
pytest tests/test_phase1.py -v
```