from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime, timezone
from groq import Groq
import json
import re
import uuid

from shared.config import GROQ_MODEL, GROQ_API_KEY, GROQ_MAX_TOKENS, GROQ_TEMPERATURE

app = FastAPI()
client = Groq(api_key=GROQ_API_KEY)


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

    condition_text = "\n".join([c.get('code', {}).get('text', 'Condition') for c in conditions[:5]]) or "No conditions"
    medication_list = [m.get('medicationCodeableConcept', {}).get('text', 'Medication') for m in medications]

    system_prompt = """You are a licensed medical social worker screening for social determinants of health risks at hospital discharge."""

    user_prompt = f"""Screen this patient for social determinants of health risks.

PATIENT: {patient_name}
ADDRESS: {address_text}
PHONE: {phone if phone else "Not provided"}
LIVES ALONE: {"Yes" if living_alone else "Not specified"}

CONDITIONS:
{condition_text}

MEDICATIONS: {", ".join(medication_list) if medication_list else "None"}

Analyze 6 domains: Transportation, Food Security, Housing, Financial Toxicity, Caregiver Support, Health Literacy.

Return JSON:
{{"risk_domains": [{{"domain": "X", "risk_level": "LOW|MEDIUM|HIGH", "rationale": "why", "recommended_action": "what"}}]}}"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
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
        risk_domains = [{"domain": d, "risk_level": "LOW", "rationale": "Not enough data", "recommended_action": "Confirm at discharge"}
                       for d in ["Transportation", "Food Security", "Housing", "Financial Toxicity", "Caregiver Support", "Health Literacy"]]

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


@app.post("/mcp")
async def mcp_endpoint(request: Request):
    try:
        body = await request.json()
        method = body.get("method", "")
        msg_id = body.get("id", str(uuid.uuid4()))
        
        if method == "initialize":
            return JSONResponse({
                "jsonrpc": "2.0", "id": msg_id,
                "result": {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                          "serverInfo": {"name": "ATLAS SDOH Screener", "version": "1.0.0"}}
            })
        
        elif method == "tools/list":
            return JSONResponse({
                "jsonrpc": "2.0", "id": msg_id,
                "result": {"tools": [{
                    "name": "screen_social_determinants",
                    "description": "Screen patient for social determinants of health risks including transportation, food security, housing, financial toxicity, and caregiver support.",
                    "inputSchema": {"type": "object", "properties": {
                        "patient_fhir_context": {"type": "object", "description": "Full FHIR patient context"}
                    }, "required": ["patient_fhir_context"]}
                }]}
            })
        
        elif method == "tools/call":
            params = body.get("params", {})
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            
            if tool_name == "screen_social_determinants":
                result = screen_sdoh(arguments.get("patient_fhir_context", {}))
                return JSONResponse({"jsonrpc": "2.0", "id": msg_id, "result": {"content": [{"type": "text", "text": str(result)}]}})
            else:
                return JSONResponse({"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": f"Tool not found: {tool_name}"}})
        
        elif method == "notifications/initialized":
            return JSONResponse({"jsonrpc": "2.0", "id": msg_id, "result": {}})
        
        return JSONResponse({"jsonrpc": "2.0", "id": msg_id, "result": {}})
    
    except Exception as e:
        return JSONResponse({"jsonrpc": "2.0", "id": "error", "error": {"code": -32603, "message": str(e)}})


@app.get("/mcp")
async def mcp_get():
    return JSONResponse({"status": "ok", "protocol": "MCP", "version": "2024-11-05", "server": "ATLAS SDOH Screener"})


@app.post("/screen-sdoh")
async def screen_sdoh_endpoint(req: SDOHRequest):
    try:
        return screen_sdoh(req.patient_fhir_context)
    except Exception as e:
        return {"error": str(e), "risk_score": "UNKNOWN", "risk_domains": [], "social_work_referral_needed": False, "community_resources": COMMUNITY_RESOURCES, "care_team_flag": "ERROR", "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}


@app.get("/health")
async def health():
    return {"status": "ok", "server": "sdoh"}


@app.get("/mcp-test")
async def mcp_test():
    return {"status": "MCP mounted", "endpoint": "/mcp"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)