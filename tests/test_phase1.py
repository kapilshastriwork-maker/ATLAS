import pytest
import json
import os
from shared import config
from shared.fhir_client import FHIRClient
from shared.context import SharedContext


def test_config_loads():
    try:
        _ = config.GROQ_API_KEY
    except ValueError:
        pytest.skip("GROQ_API_KEY not set")
    _ = config.GROQ_MODEL
    _ = config.GROQ_MAX_TOKENS
    _ = config.GROQ_TEMPERATURE
    _ = config.FHIR_BASE_URL
    _ = config.APP_NAME
    _ = config.APP_VERSION


@pytest.mark.asyncio
async def test_fhir_client_get_patient():
    client = FHIRClient(config.FHIR_BASE_URL)
    patient = await client.get_patient("90272570")
    assert isinstance(patient, dict)
    assert "resourceType" in patient
    assert patient["resourceType"] == "Patient"


@pytest.mark.asyncio
async def test_fhir_full_context():
    client = FHIRClient(config.FHIR_BASE_URL)
    context = await client.get_full_context("90272570")
    assert isinstance(context, dict)
    assert "patient" in context
    assert "conditions" in context
    assert "medications" in context
    assert "labs" in context
    assert "vitals" in context
    assert "allergies" in context


def test_shared_context_creation():
    demo_path = os.path.join(os.path.dirname(__file__), "..", "demo", "sample_patient.json")
    with open(demo_path) as f:
        patient = json.load(f)

    ctx = SharedContext(patient, {"discharge_destination": "Home", "receiving_provider": "Dr. Smith"})
    result = ctx.to_dict()

    assert "patient_id" in result
    assert "patient_name" in result
    assert "patient_age" in result
    assert "primary_diagnosis" in result
    assert "preferred_language" in result
    assert "discharge_destination" in result
    assert "receiving_provider" in result
    assert "medication_flags" in result
    assert "prior_auth_required" in result
    assert "social_risk_flags" in result
    assert "follow_up_tasks" in result