from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime, timezone
from groq import Groq
import uuid

from shared.config import GROQ_MODEL, GROQ_API_KEY, GROQ_MAX_TOKENS, GROQ_TEMPERATURE

app = FastAPI()
client = Groq(api_key=GROQ_API_KEY)


class NavigatorRequest(BaseModel):
    patient_fhir_context: dict
    medication_flags: list = []
    high_risk_medications: list = []
    follow_up_plan: str = ""
    discharge_destination: str = "home"


LANGUAGE_MAP = {
    "en": "English", "es": "Spanish", "fr": "French", "zh": "Chinese",
    "ar": "Arabic", "vi": "Vietnamese", "ko": "Korean", "tl": "Tagalog", "ru": "Russian"
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

    condition_list = [c.get('code', {}).get('text', 'Medical condition') for c in conditions[:5]]
    conditions_text = ", ".join(condition_list) or "your medical condition"
    medication_list = [m.get('medicationCodeableConcept', {}).get('text', m.get('medication', {}).get('display', 'Medication')) for m in medications]
    meds_text = "\n".join([f"- {med}" for med in medication_list]) or "No medications"
    high_risk_text = "\n".join([f"- {med}: WARNING" for med in high_risk_medications]) if high_risk_medications else ""
    allergies_text = ", ".join([a.get('code', {}).get('text', 'Allergy') for a in allergies[:5]]) or "No known allergies"
    follow_up_text = follow_up_plan or "Follow up with your doctor in 7-14 days"

    system_prompt = """You are a patient education specialist. Write discharge instructions at 6th grade level. Use simple words. Avoid jargon. Be warm."""

    language_instruction = "" if preferred_lang_code == "en" else f"IMPORTANT: Write the ENTIRE response in {display_language}. No English."

    user_prompt = f"""Write patient-friendly discharge instructions.

PATIENT: {patient_name}
DIAGNOSIS: {conditions_text}
YOUR MEDICATIONS:
{meds_text}
{f"WARNING: {high_risk_text}" if high_risk_text else ""}
ALLERGIES: {allergies_text}
FOLLOW-UP: {follow_up_text}
DISCHARGE TO: {discharge_destination}

{language_instruction}

Include:
1. Warm greeting to {patient_first_name}
2. What happened (plain language)
3. Medications (what each does, when to take)
4. First week day-by-day
5. When to call doctor
6. When to go to ER
7. Follow-up appointments
8. 3 most important things (bold)"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            max_tokens=GROQ_MAX_TOKENS,
            temperature=GROQ_TEMPERATURE
        )
        instructions = response.choices[0].message.content
    except Exception as e:
        instructions = f"Error: {str(e)}"

    key_warnings = ["Take medications as prescribed", "Keep follow-up appointments", "Call if symptoms worsen"]
    if high_risk_medications:
        key_warnings.append(f"Extra caution: {', '.join(high_risk_medications[:2])}")

    return {
        "instructions": instructions,
        "language": display_language,
        "reading_level": "6th grade",
        "key_warnings": key_warnings,
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
                "result": {"protocolVersion": "2024-11-05", "capabilities": {
                        "tools": {},
                        "extensions": {
                            "ai.promptopinion/fhir-context": {
                                "scopes": [
                                    {"name": "patient/Patient.rs", "required": True},
                                    {"name": "patient/Condition.rs", "required": False},
                                    {"name": "patient/MedicationRequest.rs", "required": False},
                                    {"name": "patient/Observation.rs", "required": False},
                                    {"name": "patient/AllergyIntolerance.rs", "required": False}
                                ]
                            }
                        }
                    },
                          "serverInfo": {"name": "ATLAS Patient Navigator", "version": "1.0.0"}}
            })
        
        elif method == "tools/list":
            return JSONResponse({
                "jsonrpc": "2.0", "id": msg_id,
                "result": {"tools": [{
                    "name": "generate_patient_instructions",
                    "description": "Generate personalized discharge instructions in patient preferred language at 6th grade level. Includes medication guidance, warning signs, and follow-up timeline.",
                    "inputSchema": {"type": "object", "properties": {
                        "patient_fhir_context": {"type": "object"},
                        "medication_flags": {"type": "array"},
                        "high_risk_medications": {"type": "array"},
                        "follow_up_plan": {"type": "string"},
                        "discharge_destination": {"type": "string"}
                    }, "required": ["patient_fhir_context"]}
                }]}
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
            
            if tool_name == "generate_patient_instructions":
                result = generate_instructions(
                    arguments.get("patient_fhir_context", {}),
                    arguments.get("medication_flags", []),
                    arguments.get("high_risk_medications", []),
                    arguments.get("follow_up_plan", ""),
                    arguments.get("discharge_destination", "home")
                )
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
    return JSONResponse({"status": "ok", "protocol": "MCP", "version": "2024-11-05", "server": "ATLAS Patient Navigator"})


@app.post("/generate-instructions")
async def generate_instructions_endpoint(req: NavigatorRequest):
    try:
        return generate_instructions(req.patient_fhir_context, req.medication_flags, req.high_risk_medications, req.follow_up_plan, req.discharge_destination)
    except Exception as e:
        return {"error": str(e), "instructions": "", "language": "English", "reading_level": "6th grade", "key_warnings": [], "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}


@app.get("/health")
async def health():
    return {"status": "ok", "server": "navigator"}


@app.get("/mcp-test")
async def mcp_test():
    return {"status": "MCP mounted", "endpoint": "/mcp"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)