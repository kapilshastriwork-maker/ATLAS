import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

from shared.fhir_client import FHIRClient
from shared.context import SharedContext
from shared.config import GROQ_API_KEY, FHIR_BASE_URL
from agents.loop_closure import LoopClosure


class ATLASOrchestrator:
    def __init__(self):
        self.fhir_client = FHIRClient(base_url=FHIR_BASE_URL)
        self.http_client = httpx.AsyncClient(timeout=60.0)
        self.mcp_servers = {
            "handoff": "http://localhost:8001/generate-handoff",
            "medication": "http://localhost:8002/reconcile-medications",
            "prior_auth": "http://localhost:8003/draft-prior-auth",
            "navigator": "http://localhost:8004/generate-instructions",
            "sdoh": "http://localhost:8005/screen-sdoh"
        }

    async def execute_transition(self, patient_id: str, transition_params: dict) -> dict:
        start_time = datetime.now(timezone.utc)
        atlas_run_id = str(uuid.uuid4())

        if patient_id.startswith("local-"):
            patient_path = Path(__file__).parent.parent / "demo" / "sample_patient.json"
            with open(patient_path) as f:
                patient_data = json.load(f)

            fhir_context = {
                "patient": patient_data,
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
            print(f"ATLAS: Loaded local demo patient from {patient_path}")
        else:
            print(f"ATLAS: Fetching FHIR context for patient {patient_id}...")
            fhir_context = await self.fhir_client.get_full_context(patient_id)

        shared_ctx = SharedContext(fhir_context, transition_params)
        print(f"ATLAS: Patient context loaded — {shared_ctx.patient_name}")

        print("ATLAS: Wave 1 — Launching Handoff, Medication, SDOH agents...")
        wave1_tasks = [
            self._call_handoff_agent(fhir_context, transition_params),
            self._call_medication_agent(fhir_context),
            self._call_sdoh_agent(fhir_context)
        ]
        wave1_results = await asyncio.gather(*wave1_tasks)
        handoff_result, medication_result, sdoh_result = wave1_results
        print("ATLAS: Wave 1 complete.")

        shared_ctx.medication_flags = medication_result.get("flags", [])
        shared_ctx.prior_auth_required = medication_result.get("prior_auth_required", [])
        shared_ctx.high_risk_medications = medication_result.get("high_risk_medications", [])
        shared_ctx.social_risk_flags = sdoh_result.get("risk_domains", [])
        print("ATLAS: Context enriched with Wave 1 findings.")

        print("ATLAS: Wave 2 — Launching Prior Auth and Navigator agents...")
        wave2_tasks = [
            self._call_prior_auth_agent(fhir_context, shared_ctx.prior_auth_required),
            self._call_navigator_agent(fhir_context, shared_ctx)
        ]
        wave2_results = await asyncio.gather(*wave2_tasks)
        prior_auth_result, navigator_result = wave2_results
        print("ATLAS: Wave 2 complete.")

        discharge_timestamp = datetime.now(timezone.utc).isoformat()
        loop_closure_task = LoopClosure(
            patient_id=patient_id,
            patient_name=shared_ctx.patient_name,
            high_risk_medications=shared_ctx.high_risk_medications,
            preferred_language=shared_ctx.preferred_language,
            discharge_timestamp=discharge_timestamp
        )
        loop_closure_dict = loop_closure_task.to_dict()
        print("ATLAS: Loop closure scheduled for 48 hours post-discharge.")

        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()

        medications_flagged = len(shared_ctx.medication_flags)
        prior_auth_items = len(shared_ctx.prior_auth_required)
        social_risk_score = sdoh_result.get("risk_score", "UNKNOWN")

        print(f"ATLAS: Complete. Duration: {duration:.2f}s")

        return {
            "atlas_run_id": atlas_run_id,
            "patient_id": patient_id,
            "patient_name": shared_ctx.patient_name,
            "generated_at": start_time.isoformat(),
            "duration_seconds": duration,
            "outputs": {
                "clinical_handoff": handoff_result,
                "medication_report": medication_result,
                "prior_auth": prior_auth_result,
                "patient_instructions": navigator_result,
                "sdoh_assessment": sdoh_result,
                "loop_closure": loop_closure_dict
            },
            "summary": {
                "total_agents_run": 6,
                "medications_flagged": medications_flagged,
                "prior_auth_items": prior_auth_items,
                "social_risk_score": social_risk_score,
                "patient_language": shared_ctx.preferred_language,
                "handoff_specialty": transition_params.get("specialty", "primary_care")
            }
        }

    async def _call_handoff_agent(self, fhir_context: dict, transition_params: dict) -> dict:
        try:
            resp = await self.http_client.post(
                self.mcp_servers["handoff"],
                json={
                    "patient_fhir_context": fhir_context,
                    "receiving_provider_specialty": transition_params.get("specialty", "primary_care"),
                    "clinical_question": transition_params.get("clinical_question", "")
                }
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"handoff_letter": "Error generating handoff", "error": str(e)}

    async def _call_medication_agent(self, fhir_context: dict) -> dict:
        try:
            patient_age = self._extract_age(fhir_context)
            resp = await self.http_client.post(
                self.mcp_servers["medication"],
                json={
                    "patient_fhir_context": fhir_context,
                    "patient_age": patient_age
                }
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"flags": [], "high_risk_medications": [], "prior_auth_required": [], "error": str(e)}

    async def _call_prior_auth_agent(self, fhir_context: dict, medications_requiring_auth: list) -> dict:
        try:
            resp = await self.http_client.post(
                self.mcp_servers["prior_auth"],
                json={
                    "patient_fhir_context": fhir_context,
                    "medications_requiring_auth": medications_requiring_auth,
                    "services_requiring_auth": [],
                    "insurance_plan_type": "commercial"
                }
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"prior_auth_letters": [], "error": str(e)}

    async def _call_navigator_agent(self, fhir_context: dict, shared_ctx: SharedContext) -> dict:
        try:
            resp = await self.http_client.post(
                self.mcp_servers["navigator"],
                json={
                    "patient_fhir_context": fhir_context,
                    "medication_flags": shared_ctx.medication_flags,
                    "high_risk_medications": getattr(shared_ctx, "high_risk_medications", []),
                    "follow_up_plan": "",
                    "discharge_destination": shared_ctx.discharge_destination
                }
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"instructions": "Error generating instructions", "error": str(e)}

    async def _call_sdoh_agent(self, fhir_context: dict) -> dict:
        try:
            resp = await self.http_client.post(
                self.mcp_servers["sdoh"],
                json={
                    "patient_fhir_context": fhir_context
                }
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"risk_score": "UNKNOWN", "risk_domains": [], "error": str(e)}

    def _extract_age(self, fhir_context: dict) -> int:
        patient = fhir_context.get("patient", {})
        birth_date = patient.get("birthDate", "")
        if birth_date:
            try:
                birth_year = int(birth_date.split("-")[0])
                return 2026 - birth_year
            except (ValueError, IndexError):
                return 65
        return 65