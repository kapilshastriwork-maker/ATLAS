from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime, timezone
from groq import Groq
import json
import re

from shared.config import GROQ_MODEL, GROQ_API_KEY, GROQ_MAX_TOKENS, GROQ_TEMPERATURE

app = FastAPI()
client = Groq(api_key=GROQ_API_KEY)


class MedicationRequest(BaseModel):
    patient_fhir_context: dict
    patient_age: int = 65


HIGH_RISK_MEDICATIONS = ["warfarin", "apixaban", "rivaroxaban", "dabigatran", "heparin", "enoxaparin",
                      "insulin", "metformin", "digoxin", "opioid", "hydrocodone", "oxycodone", "morphine",
                      "fentanyl", "codeine", "tramadol"]


def reconcile_medications(patient_fhir_context: dict, patient_age: int = 65) -> dict:
    patient = patient_fhir_context.get("patient", {})
    medications = patient_fhir_context.get("medications", [])
    conditions = patient_fhir_context.get("conditions", [])
    allergies = patient_fhir_context.get("allergies", [])

    name_array = patient.get("name", [])
    if name_array:
        name_obj = name_array[0]
        given = " ".join(name_obj.get("given", ["Unknown"]))
        family = name_obj.get("family", "Unknown")
        patient_name = f"{given} {family}"
    else:
        patient_name = "Unknown Patient"

    medication_list = [
        m.get('medicationCodeableConcept', {}).get('text', m.get('medication', {}).get('display', 'Unknown'))
        for m in medications
    ]
    meds_text = "\n".join([f"- {med}" for med in medication_list]) or "No medications on record"

    condition_list = [
        c.get('code', {}).get('text', 'Unknown')
        for c in conditions
    ]
    conditions_text = "\n".join([f"- {cond}" for cond in condition_list]) or "No conditions on record"

    has_ckd = any("kidney" in c.lower() or "renal" in c.lower() or "ckd" in c.lower() 
                  for c in condition_list)

    system_prompt = """You are a clinical pharmacist expert reviewing medications for patient safety at hospital discharge. Analyze medications and identify safety concerns."""

    user_prompt = f"""Analyze the following patient's medications and return a JSON object with the results.

PATIENT: {patient_name}, Age: {patient_age}
Conditions: {conditions_text}

MEDICATIONS:
{meds_text}

CRITICAL RULES YOU MUST FOLLOW:
- Warfarin is ALWAYS a HIGH RISK medication. No exceptions. 
  Always include it in both high_risk_medications list AND flags list.
- Insulin is ALWAYS HIGH RISK
- Opioids are ALWAYS HIGH RISK
- Any anticoagulant is ALWAYS HIGH RISK

You MUST return ONLY a valid JSON object. 
No markdown. No code blocks. No backticks. No explanation.
Start your response with {{ and end with }}

The JSON must have these exact keys:
{{
  "reconciled_medications": [...],
  "flags": [...],
  "prior_auth_required": [...],
  "high_risk_medications": [...],
  "summary": "..."
}}

Even if you are uncertain about other medications, warfarin MUST 
appear in high_risk_medications and flags."""

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
        raw_response_text = response.choices[0].message.content
        
        print("RAW GROQ RESPONSE:", raw_response_text)

        raw_text = raw_response_text.strip()
        
        if raw_text.startswith("```"):
            lines = raw_text.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            raw_text = "\n".join(lines).strip()
        
        start = raw_text.find("{")
        end = raw_text.rfind("}") + 1
        if start != -1 and end > start:
            raw_text = raw_text[start:end]
        
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as e:
            print(f"JSON PARSE ERROR: {e}")
            print(f"ATTEMPTED TO PARSE: {raw_text}")
            parsed = {
                "reconciled_medications": [],
                "flags": [],
                "prior_auth_required": [],
                "high_risk_medications": [],
                "summary": "Could not parse medication analysis.",
                "raw_output": raw_text
            }
        
        HIGH_RISK_KEYWORDS = [
            "warfarin", "coumadin",
            "insulin", "glargine", "aspart", "lispro",
            "morphine", "oxycodone", "hydrocodone", "fentanyl",
            "apixaban", "rivaroxaban", "dabigatran",
            "digoxin", "lithium", "methotrexate"
        ]

        all_med_names = []
        for med in patient_fhir_context.get("medications", []):
            med_text = med.get("medicationCodeableConcept", {}).get("text", "").lower()
            if med_text:
                all_med_names.append(med_text)

        confirmed_high_risk = parsed.get("high_risk_medications", [])
        confirmed_flags = parsed.get("flags", [])

        normalized_flags = []
        for f in confirmed_flags:
            if isinstance(f, dict):
                normalized_flags.append(f)
            elif isinstance(f, str):
                normalized_flags.append({
                    "medication": f,
                    "severity": "HIGH",
                    "issue": f"High-risk medication detected: {f}",
                    "recommendation": "Review before discharge",
                    "notify_agents": ["prior_auth", "navigator", "handoff"]
                })

        for med_name in all_med_names:
            for keyword in HIGH_RISK_KEYWORDS:
                if keyword in med_name:
                    if not any(keyword in str(m).lower() for m in confirmed_high_risk):
                        confirmed_high_risk.append(med_name)
                    if not any(keyword in str(f.get("medication", "")).lower() for f in normalized_flags):
                        normalized_flags.append({
                            "medication": med_name,
                            "severity": "HIGH",
                            "issue": f"{med_name} is a high-risk medication requiring close monitoring",
                            "recommendation": "Ensure monitoring plan is in place before discharge",
                            "notify_agents": ["prior_auth", "navigator", "handoff"]
                        })

        parsed["high_risk_medications"] = confirmed_high_risk
        parsed["flags"] = normalized_flags

        PRIOR_AUTH_TRIGGERS = [
            "warfarin", "apixaban", "rivaroxaban", "dabigatran",
            "insulin", "humira", "adalimumab", "etanercept"
        ]
        prior_auth_list = parsed.get("prior_auth_required", [])
        for med_name in all_med_names:
            for keyword in PRIOR_AUTH_TRIGGERS:
                if keyword in med_name:
                    if not any(keyword in str(p).lower() for p in prior_auth_list):
                        prior_auth_list.append(med_name)
        parsed["prior_auth_required"] = prior_auth_list

        return {
            "reconciled_medications": parsed.get("reconciled_medications", []),
            "flags": parsed.get("flags", []),
            "prior_auth_required": parsed.get("prior_auth_required", []),
            "high_risk_medications": parsed.get("high_risk_medications", []),
            "summary": parsed.get("summary", ""),
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        }
    except Exception as e:
        return {
            "reconciled_medications": [],
            "flags": [],
            "prior_auth_required": [],
            "high_risk_medications": [],
            "summary": f"Error: {str(e)}",
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        }


@app.post("/reconcile-medications")
async def reconcile_medications_endpoint(req: MedicationRequest):
    try:
        return reconcile_medications(req.patient_fhir_context, req.patient_age)
    except Exception as e:
        return {"error": str(e), "reconciled_medications": [], "flags": [], "prior_auth_required": [], "high_risk_medications": [], "summary": "", "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}


@app.get("/health")
async def health():
    return {"status": "ok", "server": "medication"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)