from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime, timezone
from groq import Groq
import json
import uuid

from shared.config import GROQ_MODEL, GROQ_API_KEY, GROQ_MAX_TOKENS, GROQ_TEMPERATURE

app = FastAPI()
client = Groq(api_key=GROQ_API_KEY)


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
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "ATLAS Clinical Handoff", "version": "1.0.0"}
                }
            })
        
        elif method == "tools/list":
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "tools": [{
                        "name": "generate_clinical_handoff",
                        "description": "Generate a tailored clinical handoff letter from FHIR patient data, customized for the receiving provider specialty. Used during hospital discharge to communicate patient status to the receiving care provider.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "patient_fhir_context": {"type": "object", "description": "Full FHIR patient context"},
                                "receiving_provider_specialty": {"type": "string", "description": "Specialty of receiving provider"},
                                "clinical_question": {"type": "string", "description": "Specific clinical question"}
                            },
                            "required": ["patient_fhir_context", "receiving_provider_specialty"]
                        }
                    }]
                }
            })
        
        elif method == "tools/call":
            params = body.get("params", {})
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            
            if tool_name == "generate_clinical_handoff":
                result = await generate_handoff(
                    arguments.get("patient_fhir_context", {}),
                    arguments.get("receiving_provider_specialty", "primary_care"),
                    arguments.get("clinical_question", "")
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
        
        else:
            return JSONResponse({"jsonrpc": "2.0", "id": msg_id, "result": {}})
    
    except Exception as e:
        return JSONResponse({"jsonrpc": "2.0", "id": "error", "error": {"code": -32603, "message": str(e)}})


@app.get("/mcp")
async def mcp_get():
    return JSONResponse({
        "status": "ok",
        "protocol": "MCP",
        "version": "2024-11-05",
        "server": "ATLAS Clinical Handoff"
    })


@app.post("/generate-handoff")
async def generate_handoff_endpoint(req: HandoffRequest):
    try:
        return generate_handoff(req.patient_fhir_context, req.receiving_provider_specialty, req.clinical_question)
    except Exception as e:
        return {"error": str(e), "handoff_letter": "", "specialty": req.receiving_provider_specialty, "patient_name": "Unknown", "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}


@app.get("/health")
async def health():
    return {"status": "ok", "server": "handoff"}


@app.get("/mcp-test")
async def mcp_test():
    return {"status": "MCP mounted", "endpoint": "/mcp"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)