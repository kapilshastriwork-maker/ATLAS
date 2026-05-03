from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime, timezone
from groq import Groq
from mcp.server.fastmcp import FastMCP

from shared.config import GROQ_MODEL, GROQ_API_KEY, GROQ_MAX_TOKENS, GROQ_TEMPERATURE

app = FastAPI()
client = Groq(api_key=GROQ_API_KEY)
mcp_server = FastMCP("ATLAS Patient Navigator")


class NavigatorRequest(BaseModel):
    patient_fhir_context: dict
    medication_flags: list = []
    high_risk_medications: list = []
    follow_up_plan: str = ""
    discharge_destination: str = "home"


LANGUAGE_MAP = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "zh": "Chinese",
    "ar": "Arabic",
    "vi": "Vietnamese",
    "ko": "Korean",
    "tl": "Tagalog",
    "ru": "Russian"
}


def get_preferred_language(patient: dict) -> str:
    comms = patient.get("communication", [])
    for comm in comms:
        if comm.get("preferred", False):
            codings = comm.get("language", {}).get("coding", [])
            if codings:
                return codings[0].get("code", "en")
    if comms:
        first_comm = comms[0]
        codings = first_comm.get("language", {}).get("coding", [])
        if codings:
            return codings[0].get("code", "en")
    return "en"


def generate_instructions(patient_fhir_context: dict, medication_flags: list = [], 
                    high_risk_medications: list = [], follow_up_plan: str = "",
                    discharge_destination: str = "home") -> dict:
    patient = patient_fhir_context.get("patient", {})
    conditions = patient_fhir_context.get("conditions", [])
    medications = patient_fhir_context.get("medications", [])
    allergies = patient_fhir_context.get("allergies", [])

    name_array = patient.get("name", [])
    if name_array:
        name_obj = name_array[0]
        given = name_obj.get("given", ["Patient"])
        patient_first_name = given[0] if given else "Patient"
        family = name_obj.get("family", "Unknown")
        patient_name = f"{' '.join(given) if given else 'Patient'} {family}"
    else:
        patient_first_name = "Patient"
        patient_name = "Unknown Patient"

    preferred_lang_code = get_preferred_language(patient)
    display_language = LANGUAGE_MAP.get(preferred_lang_code, "English")

    condition_list = [
        c.get('code', {}).get('text', 'Medical condition')
        for c in conditions[:5]
    ]
    conditions_text = ", ".join(condition_list) or "your medical condition"

    medication_list = [
        m.get('medicationCodeableConcept', {}).get('text', m.get('medication', {}).get('display', 'Medication'))
        for m in medications
    ]
    meds_text = "\n".join([f"- {med}" for med in medication_list]) or "No medications"

    high_risk_text = "\n".join([f"- {med}: IMPORTANT WARNING" for med in high_risk_medications]) if high_risk_medications else ""

    allergies_text = ", ".join([
        a.get('code', {}).get('text', 'Allergy') 
        for a in allergies[:5]
    ]) or "No known allergies"

    follow_up_text = follow_up_plan or "Follow up with your doctor in 7-14 days"

    system_prompt = """You are a patient education specialist. Write discharge instructions at a 6th grade reading level. Use simple words. Avoid medical jargon. Be warm and encouraging."""

    language_instruction = "" if preferred_lang_code == "en" else f"IMPORTANT: Write the ENTIRE response in {display_language}. Do not use any English."

    user_prompt = f"""Write patient-friendly discharge instructions.

PATIENT: {patient_name}
DIAGNOSIS: {conditions_text}
YOUR MEDICATIONS:
{meds_text}
{f"SPECIAL WARNINGS FOR HIGH-RISK MEDICATIONS:\n{high_risk_text}" if high_risk_text else ""}
ALLERGIES TO AVOID: {allergies_text}
YOUR FOLLOW-UP PLAN: {follow_up_text}
DISCHARGE DESTINATION: {discharge_destination}

{language_instruction}

Include all of these sections:
1. A warm greeting addressed to {patient_first_name}
2. What happened to you (plain language explanation)
3. Your medications (what each does and when to take it), with safety warnings for high-risk medications
4. Day-by-day timeline for your first week
5. When to call your doctor (specific symptoms to watch for)
6. When to go to the Emergency Room immediately (specific serious symptoms)
7. Your follow-up appointments
8. The 3 most important things to remember (bold these)

Write in plain language at 6th grade reading level."""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=GROQ_MAX_TOKENS,
            temperature=GROQ_TEMPERATURE
        )
        instructions = response.choices[0].message.content
    except Exception as e:
        instructions = f"Error generating instructions: {str(e)}"

    key_warnings = [
        "Take medications exactly as prescribed",
        "Keep all follow-up appointments",
        "Call doctor if symptoms worsen"
    ]
    if high_risk_medications:
        key_warnings.append(f"Extra caution with: {', '.join(high_risk_medications[:2])}")

    return {
        "instructions": instructions,
        "language": display_language,
        "reading_level": "6th grade",
        "key_warnings": key_warnings,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    }


@app.post("/generate-instructions")
async def generate_instructions_endpoint(req: NavigatorRequest):
    try:
        return generate_instructions(
            req.patient_fhir_context,
            req.medication_flags,
            req.high_risk_medications,
            req.follow_up_plan,
            req.discharge_destination
        )
    except Exception as e:
        return {"error": str(e), "instructions": "", "language": "English", "reading_level": "6th grade", "key_warnings": [], "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}


@app.get("/health")
async def health():
    return {"status": "ok", "server": "navigator"}


@app.get("/mcp-test")
async def mcp_test():
    return {"status": "MCP mounted", "endpoint": "/mcp"}


@mcp_server.tool()
async def generate_patient_instructions(
    patient_fhir_context: dict,
    medication_flags: list = [],
    high_risk_medications: list = [],
    follow_up_plan: str = "",
    discharge_destination: str = "home"
) -> dict:
    """Generate personalized discharge instructions in the 
    patient's preferred language at 6th grade reading level. 
    Includes medication guidance, warning signs, and follow-up 
    timeline."""
    return generate_instructions(
        patient_fhir_context,
        medication_flags,
        high_risk_medications,
        follow_up_plan,
        discharge_destination
    )


# Mount MCP server to FastAPI app
mcp_app = mcp_server.streamable_http_app()
app.mount("/mcp", mcp_app)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)