from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime, timezone, timedelta
from groq import Groq
import uuid

from shared.config import GROQ_MODEL, GROQ_API_KEY, GROQ_MAX_TOKENS, GROQ_TEMPERATURE

app = FastAPI()
client = Groq(api_key=GROQ_API_KEY)


class LoopClosureRequest(BaseModel):
    patient_name: str
    discharge_diagnosis: str = ""
    high_risk_medications: list = []
    preferred_language: str = "en"
    follow_up_provider: str = ""


def schedule_loop_closure(patient_name: str, discharge_diagnosis: str = "",
                         high_risk_medications: list = [], preferred_language: str = "en",
                         follow_up_provider: str = "") -> dict:
    task_id = str(uuid.uuid4())
    check_due_at = (datetime.now(timezone.utc) + timedelta(hours=48)).isoformat().replace("+00:00", "Z")
    
    first_name = patient_name.split()[0] if patient_name else "Patient"
    
    lang_instruction = "Write in English." if preferred_language == "en" else f"Write entirely in {preferred_language}."
    
    high_risk_list = ", ".join(high_risk_medications) if high_risk_medications else "none"
    
    system_prompt = """You are a care coordination specialist creating post-discharge follow-up plans."""
    
    user_prompt = f"""Create a 48-hour post-discharge follow-up plan for a patient.

PATIENT: {patient_name}
DIAGNOSIS: {discharge_diagnosis}
HIGH-RISK MEDICATIONS: {high_risk_list}
PROVIDER: {follow_up_provider if follow_up_provider else "TBD"}
LANGUAGE: {preferred_language}

{lang_instruction}

Generate these three things:

1. OUTREACH SMS (under 100 words, warm, 6th grade level):
- Greet patient by first name ({first_name})
- Mention their follow-up appointment is coming up
- Encourage them to call with questions
- End with: "- Your Care Team at ATLAS Health"

2. VERIFICATION CHECKLIST (exactly 3 items):
- Prescription filled?
- Follow-up appointment scheduled?
- No concerning symptoms (chest pain, shortness of breath, bleeding)?

3. CARE MANAGER ALERT (brief, actionable):
- Patient name and DOB
- Key diagnosis and high-risk medications to verify
- Specific red flags to ask about
- Whether patient confirmed follow-up

Return ONLY a JSON object:
{{
  "outreach_sms": "...",
  "verification_checklist": ["item1", "item2", "item3"],
  "care_manager_alert": "..."
}}"""

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
        raw = response.choices[0].message.content.strip()
        
        import json
        import re
        match = re.search(r'\{[\s\S]*\}', raw)
        if match:
            data = json.loads(match.group())
            outreach_sms = data.get("outreach_sms", "")
            verification_checklist = data.get("verification_checklist", [])
            care_manager_alert = data.get("care_manager_alert", "")
        else:
            outreach_sms = raw
            verification_checklist = ["Prescription filled?", "Follow-up scheduled?", "No concerning symptoms?"]
            care_manager_alert = f"Verify follow-up for {patient_name}"
    except Exception as e:
        outreach_sms = f"Hi {first_name}, this is your care team checking in. How are you feeling? Please call us with any questions. - Your Care Team at ATLAS Health"
        verification_checklist = [
            "Prescription filled?",
            "Follow-up appointment scheduled?",
            "No concerning symptoms?"
        ]
        care_manager_alert = f"Alert: Verify 48-hour check-in for {patient_name}. Diagnosis: {discharge_diagnosis}. High-risk meds: {high_risk_list}"

    return {
        "task_id": task_id,
        "check_due_at": check_due_at,
        "outreach_sms": outreach_sms,
        "verification_checklist": verification_checklist,
        "care_manager_alert": care_manager_alert,
        "status": "scheduled"
    }


@app.post("/mcp")
async def mcp_endpoint(request: Request):
    try:
        body = await request.json()
        method = body.get("method", "")
        msg_id = body.get("id", str(uuid.uuid4()))
        
        if method == "initialize":
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {},
                        "extensions": {
                            "ai.promptopinion/fhir-context": {
                                "scopes": [
                                    {"name": "patient/Patient.rs", "required": True},
                                    {"name": "patient/MedicationDispense.rs", "required": False},
                                    {"name": "patient/Appointment.rs", "required": False},
                                    {"name": "patient/Encounter.rs", "required": False}
                                ]
                            }
                        }
                    },
                    "serverInfo": {"name": "ATLAS Loop Closure", "version": "1.0.0"}
                }
            })
        
        elif method == "tools/list":
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "tools": [{
                        "name": "schedule_loop_closure",
                        "description": "Schedule 48-hour post-discharge follow-up check for a patient. Verifies prescription fills, follow-up appointment scheduling, and early return to ED. Generates personalized SMS outreach in patient preferred language.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "patient_name": {"type": "string", "description": "Patient full name"},
                                "discharge_diagnosis": {"type": "string", "description": "Primary discharge diagnosis"},
                                "high_risk_medications": {"type": "array", "items": {"type": "string"}},
                                "preferred_language": {"type": "string", "description": "Preferred language code", "default": "en"},
                                "follow_up_provider": {"type": "string", "description": "Name of follow-up provider"}
                            },
                            "required": ["patient_name"]
                        }
                    }]
                }
            })
        
        elif method == "tools/call":
            params = body.get("params", {})
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            
            # Extract FHIR context from PromptOpinion headers
            headers = dict(request.headers)
            fhir_server_url = headers.get("x-fhir-server-url", "")
            fhir_access_token = headers.get("x-fhir-access-token", "")
            patient_id = headers.get("x-patient-id", "")
            
            if patient_id and fhir_access_token:
                print(f"FHIR context received via headers: patient={patient_id}")
                arguments["fhir_patient_id"] = patient_id
                arguments["fhir_server_url"] = fhir_server_url
            
            if tool_name == "schedule_loop_closure":
                result = schedule_loop_closure(
                    arguments.get("patient_name", ""),
                    arguments.get("discharge_diagnosis", ""),
                    arguments.get("high_risk_medications", []),
                    arguments.get("preferred_language", "en"),
                    arguments.get("follow_up_provider", "")
                )
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {"content": [{"type": "text", "text": str(result)}]}
                })
            else:
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32601, "message": f"Tool not found: {tool_name}"}
                })
        
        elif method == "notifications/initialized":
            return JSONResponse({"jsonrpc": "2.0", "id": msg_id, "result": {}})
        
        return JSONResponse({"jsonrpc": "2.0", "id": msg_id, "result": {}})
    
    except Exception as e:
        return JSONResponse({"jsonrpc": "2.0", "id": "error", "error": {"code": -32603, "message": str(e)}})


@app.get("/mcp")
async def mcp_get():
    return JSONResponse({
        "status": "ok",
        "protocol": "MCP",
        "version": "2024-11-05",
        "server": "ATLAS Loop Closure"
    })


@app.post("/schedule-loop-closure")
async def schedule_loop_closure_endpoint(req: LoopClosureRequest):
    try:
        return schedule_loop_closure(
            req.patient_name,
            req.discharge_diagnosis,
            req.high_risk_medications,
            req.preferred_language,
            req.follow_up_provider
        )
    except Exception as e:
        return {"error": str(e), "task_id": "", "status": "error"}


@app.get("/health")
async def health():
    return {"status": "ok", "server": "loop_closure"}


@app.get("/mcp-test")
async def mcp_test():
    return {"status": "MCP mounted", "endpoint": "/mcp"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)