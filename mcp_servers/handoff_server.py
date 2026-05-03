from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime, timezone
from groq import Groq
import json
from mcp.server.fastmcp import FastMCP

from shared.config import GROQ_MODEL, GROQ_API_KEY, GROQ_MAX_TOKENS, GROQ_TEMPERATURE

app = FastAPI()
client = Groq(api_key=GROQ_API_KEY)
mcp = FastMCP("ATLAS Clinical Handoff")


class HandoffRequest(BaseModel):
    patient_fhir_context: dict
    receiving_provider_specialty: str
    clinical_question: str = ""


SPECIALTY_INSTRUCTIONS = {
    "cardiology": "Focus on ejection fraction, BNP trends, fluid balance, rhythm findings, hemodynamic status. Lead with cardiac diagnosis.",
    "primary_care": "Comprehensive summary. Emphasize chronic disease management, follow-up needs, patient education gaps.",
    "nephrology": "Lead with GFR trend, creatinine trajectory, electrolyte status, medication adjustments for renal dosing.",
    "neurology": "Focus on neurological exam findings, imaging results, cognitive or functional changes.",
    "pulmonology": "Focus on oxygen requirements, ABG results, ventilator settings if applicable, respiratory status.",
    "oncology": "Lead with cancer staging, treatment plan, chemotherapy/immunotherapy details, prognosis.",
    "default": "Provide a comprehensive clinical summary."
}


def generate_handoff(patient_fhir_context: dict, specialty: str, clinical_question: str = "") -> dict:
    patient = patient_fhir_context.get("patient", {})
    conditions = patient_fhir_context.get("conditions", [])
    medications = patient_fhir_context.get("medications", [])
    allergies = patient_fhir_context.get("allergies", [])

    name_array = patient.get("name", [])
    if name_array:
        name_obj = name_array[0]
        given = " ".join(name_obj.get("given", ["Unknown"]))
        family = name_obj.get("family", "Unknown")
        patient_name = f"{given} {family}"
    else:
        patient_name = "Unknown Patient"

    specialty_instructions = SPECIALTY_INSTRUCTIONS.get(specialty.lower(), SPECIALTY_INSTRUCTIONS["default"])

    conditions_text = "\n".join([
        f"- {c.get('code', {}).get('text', 'Unknown condition')}"
        for c in conditions[:10]
    ]) or "No conditions on record"

    medications_text = "\n".join([
        f"- {m.get('medicationCodeableConcept', {}).get('text', m.get('medication', {}).get('display', 'Unknown medication'))}"
        for m in medications[:15]
    ]) or "No medications on record"

    allergies_text = "\n".join([
        f"- {a.get('code', {}).get('text', a.get('substance', {}).get('text', 'Unknown'))} ({a.get('reaction', [{}])[0].get('severity', 'Unknown')})"
        for a in allergies[:10]
    ]) or "No known allergies"

    system_prompt = """You are a senior attending physician writing a clinical handoff letter. Write in professional clinical language. Be concise but complete. A clinician reading this should have everything they need within 60 seconds."""

    user_prompt = f"""Write a clinical handoff letter for a patient being transferred to {specialty} care.

{specialty_instructions}

PATIENT INFORMATION:
- Name: {patient_name}
- Patient ID: {patient.get('id', 'Unknown')}

CLINICAL CONDITIONS:
{conditions_text}

CURRENT MEDICATIONS:
{medications_text}

ALLERGIES:
{allergies_text}

CLINICAL QUESTION FROM REFERRING PROVIDER:
{clinical_question if clinical_question else "None provided"}

Please include these exact sections in your handoff letter:
1. Patient identification and reason for transfer
2. Hospital course summary
3. Specialty-relevant clinical findings
4. Current medications with changes
5. Outstanding issues and pending results
6. Specific questions for the receiving provider
7. Follow-up plan and timeline

Return ONLY the handoff letter text. No markdown formatting. No JSON."""

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
        handoff_letter = response.choices[0].message.content
    except Exception as e:
        handoff_letter = f"Error generating handoff: {str(e)}"

    return {
        "handoff_letter": handoff_letter,
        "specialty": specialty,
        "patient_name": patient_name,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    }


@mcp.tool()
async def generate_clinical_handoff(
    patient_fhir_context: dict,
    receiving_provider_specialty: str,
    clinical_question: str = ""
) -> dict:
    """Generate a tailored clinical handoff letter from FHIR 
    patient data, customized for the receiving provider specialty.
    Used during hospital discharge to communicate patient status
    to the receiving care provider."""
    return generate_handoff(
        patient_fhir_context, 
        receiving_provider_specialty, 
        clinical_question
    )


@app.post("/generate-handoff")
async def generate_handoff_endpoint(req: HandoffRequest):
    try:
        return generate_handoff(req.patient_fhir_context, req.receiving_provider_specialty, req.clinical_question)
    except Exception as e:
        return {"error": str(e), "handoff_letter": "", "specialty": req.receiving_provider_specialty, "patient_name": "Unknown", "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}


@app.get("/health")
async def health():
    return {"status": "ok", "server": "handoff"}


# Mount MCP server to FastAPI app
app.mount("/mcp", mcp.streamable_http_app())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)