from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime, timezone
from groq import Groq
import json
import re

from shared.config import GROQ_MODEL, GROQ_API_KEY, GROQ_MAX_TOKENS, GROQ_TEMPERATURE

app = FastAPI()
client = Groq(api_key=GROQ_API_KEY)


class PriorAuthRequest(BaseModel):
    patient_fhir_context: dict
    medications_requiring_auth: list = []
    services_requiring_auth: list = []
    insurance_plan_type: str = "commercial"


def draft_prior_auth(patient_fhir_context: dict, medications_requiring_auth: list = [], 
                   services_requiring_auth: list = [], insurance_plan_type: str = "commercial") -> dict:
    patient = patient_fhir_context.get("patient", {})
    conditions = patient_fhir_context.get("conditions", [])
    labs = patient_fhir_context.get("labs", [])
    vitals = patient_fhir_context.get("vitals", [])

    name_array = patient.get("name", [])
    if name_array:
        name_obj = name_array[0]
        given = " ".join(name_obj.get("given", ["Unknown"]))
        family = name_obj.get("family", "Unknown")
        patient_name = f"{given} {family}"
    else:
        patient_name = "Unknown Patient"

    birth_date = patient.get("birthDate", "Unknown")

    primary_condition = conditions[0].get("code", {}).get("text", "Relevant medical condition") if conditions else "Relevant medical condition"
    
    condition_text = "\n".join([
        f"- {c.get('code', {}).get('text', 'Condition')}"
        for c in conditions[:5]
    ]) or "No conditions documented"

    labs_text = "\n".join([
        f"- {l.get('code', {}).get('text', 'Lab')}: {l.get('value', {}).get('value', 'N/A')} {l.get('value', {}).get('unit', '')}"
        for l in labs[:10]
    ]) or "No recent labs"

    vitals_text = "\n".join([
        f"- {v.get('code', {}).get('text', 'Vital')}: {v.get('value', {}).get('value', 'N/A')} {v.get('value', {}).get('unit', '')}"
        for v in vitals[:5]
    ]) or "No vitals on record"

    all_items = medications_requiring_auth + services_requiring_auth

    if not all_items:
        return {
            "prior_auth_letters": [],
            "total_items": 0,
            "note": "No medications or services require prior authorization",
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        }

    prior_auth_letters = []

    for item in all_items:
        urgency = "urgent" if any(keyword in item.lower() for keyword in ["warfarin", "insulin", "chemo", "biologic", "emergency"]) else "routine"
        estimated_days = 2 if urgency == "urgent" else 7

        system_prompt = """You are a healthcare prior authorization specialist with 15 years of experience. Write authorization requests that anticipate denial criteria and address them with clinical evidence."""

        user_prompt = f"""Write a prior authorization letter for: {item}

PATIENT INFORMATION:
- Name: {patient_name}
- Date of Birth: {birth_date}

PRIMARY DIAGNOSIS:
{primary_condition}

ALL CONDITIONS:
{condition_text}

RELEVANT CLINICAL DATA:
Labs: {labs_text}
Vitals: {vitals_text}

INSURANCE TYPE: {insurance_plan_type}

The letter must include:
1. Patient name and date of birth
2. Specific medication/service with CPT or NDC code (use placeholder [CPT CODE])
3. Primary diagnosis (use clinical name, ICD-10 placeholder)
4. Medical necessity statement tied to clinical guidelines
5. Supporting clinical evidence from the data above
6. Statement of failed alternative treatments where applicable
7. Urgency statement: {"This is clinically urgent due to risk of complication" if urgency == "urgent" else "Routine request for ongoing care"}

Return ONLY the letter text. No markdown. No JSON."""

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
            letter = response.choices[0].message.content
        except Exception as e:
            letter = f"Error generating letter: {str(e)}"

        prior_auth_letters.append({
            "item_name": item,
            "letter": letter,
            "urgency": urgency,
            "estimated_approval_days": estimated_days
        })

    return {
        "prior_auth_letters": prior_auth_letters,
        "total_items": len(prior_auth_letters),
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    }


@app.post("/draft-prior-auth")
async def draft_prior_auth_endpoint(req: PriorAuthRequest):
    try:
        return draft_prior_auth(
            req.patient_fhir_context,
            req.medications_requiring_auth,
            req.services_requiring_auth,
            req.insurance_plan_type
        )
    except Exception as e:
        return {"error": str(e), "prior_auth_letters": [], "total_items": 0, "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}


@app.get("/health")
async def health():
    return {"status": "ok", "server": "prior_auth"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)