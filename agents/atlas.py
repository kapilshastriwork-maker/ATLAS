import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

from agents.orchestrator import ATLASOrchestrator


async def run_atlas(patient_id: str, transition_params: dict) -> dict:
    orchestrator = ATLASOrchestrator()
    result = await orchestrator.execute_transition(patient_id, transition_params)

    output_path = Path(__file__).parent.parent / "demo" / "atlas_output.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    summary = result.get("summary", {})
    outputs = result.get("outputs", {})

    print("\n" + "=" * 50)
    print("ATLAS Phase 3 - Care Transition Complete")
    print("=" * 50)

    handoff = outputs.get("clinical_handoff", {})
    specialty = handoff.get("specialty", "unknown")
    print(f"[OK] Clinical Handoff Generated ({specialty})")

    med = outputs.get("medication_report", {})
    flagged = summary.get("medications_flagged", 0)
    high_risk = len(med.get("high_risk_medications", []))
    print(f"[OK] Medications Analyzed ({flagged} flagged, {high_risk} high-risk)")

    prior_auth = outputs.get("prior_auth", {})
    items = summary.get("prior_auth_items", 0)
    print(f"[OK] Prior Auth Drafted ({items} items)")

    nav = outputs.get("patient_instructions", {})
    lang = summary.get("patient_language", "unknown")
    print(f"[OK] Patient Instructions Generated ({lang})")

    sdoh = outputs.get("sdoh_assessment", {})
    risk = summary.get("social_risk_score", "unknown")
    print(f"[OK] SDOH Assessment Complete ({risk} risk)")

    loop = outputs.get("loop_closure", {})
    due = loop.get("check_due_at", "unknown")
    print(f"[OK] Loop Closure Scheduled (check due at: {due})")

    duration = result.get("duration_seconds", 0)
    print(f"[TIME] Total Duration: {duration:.2f} seconds")
    print("=" * 50)

    print(f"\nOutput saved to: {output_path}")

    return result


if __name__ == "__main__":
    asyncio.run(run_atlas(
        patient_id="local-maria-garcia",
        transition_params={
            "specialty": "cardiology",
            "clinical_question": "Please evaluate and manage her heart failure. Patient was admitted for acute decompensation.",
            "discharge_destination": "home with home health",
            "receiving_provider": "Dr. Cardiology Specialist"
        }
    ))