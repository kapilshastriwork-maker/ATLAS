import pytest
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_servers import handoff_server, medication_server, prior_auth_server, navigator_server, sdoh_server


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


def test_handoff_generates_output():
    result = handoff_server.generate_handoff(
        patient_fhir_context=MOCK_FHIR_CONTEXT,
        specialty="cardiology",
        clinical_question="What is the current EF?"
    )
    
    assert "handoff_letter" in result
    assert isinstance(result["handoff_letter"], str)
    assert len(result["handoff_letter"]) > 100
    assert result.get("specialty") == "cardiology"


def test_medication_flags_warfarin():
    result = medication_server.reconcile_medications(
        patient_fhir_context=MOCK_FHIR_CONTEXT,
        patient_age=68
    )
    
    high_risk = result.get("high_risk_medications", [])
    flags = result.get("flags", [])
    
    has_warfarin_flag = any(
        "warfarin" in str(med).lower() 
        for med in high_risk
    )
    has_warfarin_in_flags = any(
        "warfarin" in str(flag.get("medication", "")).lower()
        for flag in flags
    )
    
    assert has_warfarin_flag or has_warfarin_in_flags, f"Warfarin should be flagged as high risk. Got: high_risk={high_risk}, flags={flags}"


def test_navigator_generates_spanish():
    result = navigator_server.generate_instructions(
        patient_fhir_context=MOCK_FHIR_CONTEXT,
        medication_flags=[],
        high_risk_medications=["Warfarin"],
        follow_up_plan="Cardiology follow-up in 7 days",
        discharge_destination="home"
    )
    
    lang = result.get("language", "")
    assert lang == "Spanish", f"Expected Spanish, got: {lang}"


def test_sdoh_returns_risk_score():
    result = sdoh_server.screen_sdoh(
        patient_fhir_context=MOCK_FHIR_CONTEXT
    )
    
    assert "risk_score" in result
    assert result["risk_score"] in ["LOW", "MEDIUM", "HIGH"], f"Invalid risk_score: {result['risk_score']}"


def test_prior_auth_with_empty_list():
    result = prior_auth_server.draft_prior_auth(
        patient_fhir_context=MOCK_FHIR_CONTEXT,
        medications_requiring_auth=[],
        services_requiring_auth=[],
        insurance_plan_type="commercial"
    )
    
    letters = result.get("prior_auth_letters", [])
    assert isinstance(letters, list), f"prior_auth_letters should be a list, got: {type(letters)}"
    assert len(letters) == 0, f"Expected empty list for empty input, got: {letters}"