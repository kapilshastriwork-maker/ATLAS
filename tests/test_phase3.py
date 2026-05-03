import pytest
import pytest_asyncio
import asyncio
import json
import os
import sys
import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.orchestrator import ATLASOrchestrator
from agents.loop_closure import LoopClosure


def load_sample_patient():
    demo_path = os.path.join(os.path.dirname(__file__), "..", "demo", "sample_patient.json")
    with open(demo_path) as f:
        return json.load(f)


MOCK_FHIR_CONTEXT = {
    "patient": load_sample_patient(),
    "conditions": [
        {"resourceType": "Condition", 
         "code": {"text": "Heart Failure with Reduced Ejection Fraction"},
         "clinicalStatus": {"coding": [{"code": "active"}]}},
        {"resourceType": "Condition",
         "code": {"text": "Type 2 Diabetes Mellitus"},
         "clinicalStatus": {"coding": [{"code": "active"}]}},
        {"resourceType": "Condition",
         "code": {"text": "Chronic Kidney Disease Stage 3"},
         "clinicalStatus": {"coding": [{"code": "active"}]}}
    ],
    "medications": [
        {"resourceType": "MedicationRequest",
         "medicationCodeableConcept": {"text": "Warfarin 5mg daily"}},
        {"resourceType": "MedicationRequest",
         "medicationCodeableConcept": {"text": "Furosemide 40mg daily"}},
        {"resourceType": "MedicationRequest",
         "medicationCodeableConcept": {"text": "Metformin 500mg twice daily"}},
        {"resourceType": "MedicationRequest",
         "medicationCodeableConcept": {"text": "Metoprolol 25mg twice daily"}}
    ],
    "labs": [],
    "vitals": [],
    "allergies": []
}


async def check_servers_running():
    ports = [8001, 8002, 8003, 8004, 8005]
    async with httpx.AsyncClient(timeout=5.0) as client:
        for port in ports:
            try:
                resp = await client.get(f"http://localhost:{port}/health")
                if resp.status_code != 200:
                    return False
            except Exception:
                return False
    return True


@pytest_asyncio.fixture
async def servers_ready():
    if not await check_servers_running():
        pytest.skip("MCP servers not running — start with: python mcp_servers.run_all.py")
    return True


@pytest.mark.asyncio
async def test_orchestrator_full_run(servers_ready):
    orchestrator = ATLASOrchestrator()
    result = await orchestrator.execute_transition(
        "local-maria-garcia",
        {
            "specialty": "cardiology",
            "clinical_question": "Evaluate heart failure management"
        }
    )

    assert "outputs" in result
    outputs = result["outputs"]
    assert "clinical_handoff" in outputs
    assert "medication_report" in outputs
    assert "prior_auth" in outputs
    assert "patient_instructions" in outputs
    assert "sdoh_assessment" in outputs
    assert "loop_closure" in outputs

    assert result["summary"]["total_agents_run"] == 6
    assert result["duration_seconds"] < 120


def test_loop_closure_creation():
    task = LoopClosure(
        patient_id="test-123",
        patient_name="Test Patient",
        high_risk_medications=["Warfarin"],
        preferred_language="en",
        discharge_timestamp="2026-01-15T10:00:00Z"
    )
    result = task.to_dict()

    assert result["task_id"] is not None
    assert result["task_id"] != ""

    scheduled = result["scheduled_at"]
    due = result["check_due_at"]

    from datetime import datetime
    scheduled_dt = datetime.fromisoformat(scheduled.replace("Z", "+00:00"))
    due_dt = datetime.fromisoformat(due.replace("Z", "+00:00"))
    diff_hours = (due_dt - scheduled_dt).total_seconds() / 3600
    assert 47 <= diff_hours <= 49, f"Expected ~48 hours, got {diff_hours}"

    assert "prescription_fill_verification" in result["checks_to_perform"]
    assert result["outreach_message"] != ""


@pytest.mark.asyncio
async def test_orchestrator_handles_server_down(servers_ready):
    orchestrator = ATLASOrchestrator()
    orchestrator.mcp_servers["handoff"] = "http://localhost:9999/generate-handoff"

    result = await orchestrator.execute_transition(
        "local-maria-garcia",
        {"specialty": "cardiology"}
    )

    assert "error" not in result or "outputs" in result
    handoff = result.get("outputs", {}).get("clinical_handoff", {})
    assert "error" in handoff or handoff.get("handoff_letter", "").startswith("Error")