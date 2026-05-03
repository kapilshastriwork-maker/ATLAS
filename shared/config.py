import os
from dotenv import load_dotenv

load_dotenv()

_DEFAULTS = {
    "GROQ_MODEL": "llama-3.3-70b-versatile",
    "GROQ_MAX_TOKENS": "2000",
    "GROQ_TEMPERATURE": "0.3",
    "APP_NAME": "ATLAS",
    "APP_VERSION": "1.0.0"
}

def __getattr__(name):
    if name == "GROQ_API_KEY":
        key = os.getenv("GROQ_API_KEY")
        if not key or key == "your_groq_api_key_here":
            raise ValueError(f"GROQ_API_KEY is missing or still set to placeholder value. Please set a valid API key in .env")
        return key
    
    if name in _DEFAULTS or os.getenv(name):
        value = os.getenv(name, _DEFAULTS.get(name))
        if name in ("GROQ_MAX_TOKENS",):
            return int(value)
        if name in ("GROQ_TEMPERATURE",):
            return float(value)
        return value
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return ["GROQ_API_KEY", "GROQ_MODEL", "GROQ_MAX_TOKENS", "GROQ_TEMPERATURE", 
            "FHIR_BASE_URL", "APP_NAME", "APP_VERSION"]