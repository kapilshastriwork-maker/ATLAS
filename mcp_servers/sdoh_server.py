from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime, timezone
from groq import Groq
import json
import re
from mcp.server.fastmcp import FastMCP

from shared.config import GROQ_MODEL, GROQ_API_KEY, GROQ_MAX_TOKENS, GROQ_TEMPERATURE

app = FastAPI()
client = Groq(api_key=GROQ_API_KEY)
mcp = FastMCP("ATLAS SDOH Screener")


class SDOHRequest(BaseModel):
    patient_fhir_context: dict


COMMUNITY_RESOURCES = [
    {"name": "211 Helpline", "description": "Local social services referral", "contact": "Dial 2-1-1"},
    {"name": "NeedyMeds", "description": "Prescription assistance programs", "contact": "www.needymeds.org"},
    {"name": "Meals on Wheels", "description": "Home meal delivery for seniors", "contact": "www.mealsonwheelsamerica.org"}
]


def screen_sdoh(patient_fhir_context: dict) -> dict:
    patient = patient_fhir_context.get("patient", {})
    conditions = patient_fhir_context.get("conditions", [])
    medications = patient_fhir_context.get("medications", [])
    observations = patient_fhir_context.get("labs", []) + patient_fhir_context.get("vitals", [])

    name_array = patient.get("name", [])
    if name_array:
        name_obj = name_array[0]
        given = " ".join(name_obj.get("given", ["Unknown"]))
        family = name_obj.get("family", "Unknown")
        patient_name = f"{given} {family}"
    else:
        patient_name = "Unknown Patient"

    address = patient.get("address", [{}])[0] if patient.get("address") else {}
    address_text = f"{address.get('line', ['Unknown'])[0]}, {address.get('city', '')}, {address.get('state', '')} {address.get('postalCode', '')}"

    telecom = patient.get("telecom", [])
    phone = ""
    for t in telecom:
        if t.get("system") == "phone":
            phone = t.get("value", "")
            break

    living_alone = len(patient.get("address", [])) > 0

    condition_text = "\n".join([
        c.get('code', {}).get('text', 'Condition')
        for c in conditions[:5]
    ]) or "No conditions documented"

    medication_list = [
        m.get('medicationCodeableConcept', {}).get('text', 'Medication')
        for m in medications
    ]
    high_cost_meds = any(
        "insulin" in med.lower() or "biologic" in med.lower() or "specialty" in med.lower()
        for med in medication_list
    )

    system_prompt = """You are a licensed medical social worker screening for social determinants of health risks at hospital discharge."""

    user_prompt = f"""Screen this patient for social determinants of health risks.

PATIENT: {patient_name}
ADDRESS: {address_text}
PHONE: {phone if phone else "Not provided"}
LIVES ALONE: {"Yes" if living_alone else "Not specified"}

CONDITIONS:
{condition_text}

MEDICATIONS: {", ".join(medication_list) if medication_list else "None"}

Analyze these 6 domains and return a JSON object:
1. Transportation: Can patient get to follow-up appointments?
2. Food Security: Risk of food insecurity?
3. Housing: Housing stability?
4. Financial Toxicity: Can patient afford medications?
5. Caregiver Support: Does patient have help at home?
6. Health Literacy: Can patient manage their care?

Return ONLY a JSON object (no markdown, no code blocks) with this structure:
{{
  "risk_domains": [
    {{"domain": "Transportation", "risk_level": "LOW|MEDIUM|HIGH", "rationale": "why", "recommended_action": "what to do"}},
    {{"domain": "Food Security", "risk_level": "LOW|MEDIUM|HIGH", "rationale": "why", "recommended_action": "what to do"}},
    {{"domain": "Housing", "risk_level": "LOW|MEDIUM|HIGH", "rationale": "why", "recommended_action": "what to do"}},
    {{"domain": "Financial Toxicity", "risk_level": "LOW|MEDIUM|HIGH", "rationale": "why", "recommended_action": "what to do"}},
    {{"domain": "Caregiver Support", "risk_level": "LOW|MEDIUM|HIGH", "rationale": "why", "recommended_action": "what to do"}},
    {{"domain": "Health Literacy", "risk_level": "LOW|MEDIUM|HIGH", "rationale": "why", "recommended_action": "what to do"}}
  ]
}}

Consider: no phone = HIGH transportation risk, living alone = MEDIUM caregiver risk, high-cost medications = MEDIUM/HIGH financial risk.

Return ONLY the JSON."""

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
        raw_output = response.choices[0].message.content

        json_match = re.search(r'\{[\s\S]*\}', raw_output)
        if json_match:
            result = json.loads(json_match.group())
            risk_domains = result.get("risk_domains", [])
        else:
            risk_domains = []
    except Exception as e:
        risk_domains = []

    if not risk_domains:
        risk_domains = [
            {"domain": d, "risk_level": "LOW", "rationale": "Not enough data to assess", "recommended_action": "Confirm at discharge"}
            for d in ["Transportation", "Food Security", "Housing", "Financial Toxicity", "Caregiver Support", "Health Literacy"]
        ]

    high_count = sum(1 for d in risk_domains if d.get("risk_level") == "HIGH")
    medium_count = sum(1 for d in risk_domains if d.get("risk_level") == "MEDIUM")

    risk_score = "HIGH" if high_count > 0 else "MEDIUM" if medium_count >= 2 else "LOW"

    social_work_referral = high_count > 0 or medium_count >= 3

    care_team_flag = "URGENT" if risk_score == "HIGH" else "ROUTINE" if risk_score == "MEDIUM" else "NONE"

    return {
        "risk_score": risk_score,
        "risk_domains": risk_domains,
        "social_work_referral_needed": social_work_referral,
        "community_resources": COMMUNITY_RESOURCES,
        "care_team_flag": care_team_flag,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    }


@app.post("/screen-sdoh")
async def screen_sdoh_endpoint(req: SDOHRequest):
    try:
        return screen_sdoh(req.patient_fhir_context)
    except Exception as e:
        return {"error": str(e), "risk_score": "UNKNOWN", "risk_domains": [], "social_work_referral_needed": False, "community_resources": COMMUNITY_RESOURCES, "care_team_flag": "ERROR", "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}


@app.get("/health")
async def health():
    return {"status": "ok", "server": "sdoh"}


@mcp.tool()
async def screen_social_determinants(
    patient_fhir_context: dict
) -> dict:
    """Screen patient for social determinants of health risks 
    at discharge including transportation, food security, housing, 
    financial toxicity, and caregiver support. Returns risk score 
    and community resource recommendations."""
    return screen_sdoh(patient_fhir_context)


# Mount MCP server to FastAPI app
app.mount("/mcp", mcp.streamable_http_app())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)