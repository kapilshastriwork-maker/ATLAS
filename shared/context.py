from typing import Any


class SharedContext:
    def __init__(self, fhir_context: dict, transition_params: dict):
        patient = fhir_context.get("patient", fhir_context)
        self.patient_id = patient.get("id", "unknown")
        
        name_array = patient.get("name", [])
        if name_array:
            name_obj = name_array[0]
            given = " ".join(name_obj.get("given", ["Unknown"]))
            family = name_obj.get("family", "Unknown")
            self.patient_name = f"{given} {family}"
        else:
            self.patient_name = "Unknown Patient"

        birth_date = fhir_context.get("birthDate", "")
        if birth_date:
            try:
                year = int(birth_date.split("-")[0])
                self.patient_age = 2026 - year
            except (ValueError, IndexError):
                self.patient_age = None
        else:
            self.patient_age = None

        comms = patient.get("communication", [])
        preferred_lang = "en"
        for comm in comms:
            if comm.get("preferred", False):
                codings = comm.get("language", {}).get("coding", [])
                if codings:
                    preferred_lang = codings[0].get("code", "en")
                    break
        if preferred_lang == "en" and comms:
            first_comm = comms[0]
            codings = first_comm.get("language", {}).get("coding", [])
            if codings:
                preferred_lang = codings[0].get("code", "en")
        self.preferred_language = preferred_lang

        self.primary_diagnosis = ""
        self.discharge_destination = transition_params.get("discharge_destination", "")
        self.receiving_provider = transition_params.get("receiving_provider", "")

        self.medication_flags = []
        self.prior_auth_required = []
        self.social_risk_flags = []
        self.follow_up_tasks = []

    def update_from_results(self, results: list):
        for result in results:
            if isinstance(result, dict):
                if "medication_flags" in result:
                    self.medication_flags.extend(result.get("medication_flags", []))
                if "prior_auth_required" in result:
                    self.prior_auth_required.extend(result.get("prior_auth_required", []))
                if "social_risk_flags" in result:
                    self.social_risk_flags.extend(result.get("social_risk_flags", []))
                if "follow_up_tasks" in result:
                    self.follow_up_tasks.extend(result.get("follow_up_tasks", []))

    def to_dict(self) -> dict:
        return {
            "patient_id": self.patient_id,
            "patient_name": self.patient_name,
            "patient_age": self.patient_age,
            "primary_diagnosis": self.primary_diagnosis,
            "preferred_language": self.preferred_language,
            "discharge_destination": self.discharge_destination,
            "receiving_provider": self.receiving_provider,
            "medication_flags": self.medication_flags,
            "prior_auth_required": self.prior_auth_required,
            "social_risk_flags": self.social_risk_flags,
            "follow_up_tasks": self.follow_up_tasks
        }