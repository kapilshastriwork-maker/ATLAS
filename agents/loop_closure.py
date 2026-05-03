import uuid
from datetime import datetime, timezone, timedelta

from groq import Groq
from shared.config import GROQ_API_KEY, GROQ_MODEL


class LoopClosure:
    def __init__(self, patient_id: str, patient_name: str, high_risk_medications: list,
                 preferred_language: str, discharge_timestamp: str):
        self.patient_id = patient_id
        self.patient_name = patient_name
        self.high_risk_medications = high_risk_medications
        self.preferred_language = preferred_language
        self.discharge_timestamp = discharge_timestamp
        self.task_id = str(uuid.uuid4())
        self.status = "scheduled"

        discharge_dt = self._parse_timestamp(discharge_timestamp)
        check_due_dt = discharge_dt + timedelta(hours=48)
        self.check_due_at = check_due_dt.isoformat().replace("+00:00", "Z")

        self.outreach_message = self.generate_outreach_message()

    def _parse_timestamp(self, ts: str) -> datetime:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    def to_dict(self) -> dict:
        discharge_dt = self._parse_timestamp(self.discharge_timestamp)
        check_due_dt = discharge_dt + timedelta(hours=48)
        check_due = check_due_dt.isoformat().replace("+00:00", "Z")

        return {
            "task_id": self.task_id,
            "patient_id": self.patient_id,
            "patient_name": self.patient_name,
            "status": self.status,
            "scheduled_at": self.discharge_timestamp,
            "check_due_at": check_due,
            "high_risk_medications": self.high_risk_medications,
            "checks_to_perform": [
                "prescription_fill_verification",
                "follow_up_appointment_scheduled",
                "early_return_ed_check"
            ],
            "outreach_message": self.outreach_message
        }

    def generate_outreach_message(self) -> str:
        first_name = self.patient_name.split()[0] if self.patient_name else "Patient"

        if self.preferred_language == "es":
            language_instruction = "Write the entire message in Spanish."
        else:
            language_instruction = "Write in English."

        prompt = f"""Write a warm 48-hour check-in SMS message for a heart failure patient.

PATIENT FIRST NAME: {first_name}

{language_instruction}

Requirements:
- Under 100 words
- At 6th grade reading level
- Warm and encouraging, not alarming
- Mention they should call if they have questions
- Include placeholder: "Call us at [CLINIC_PHONE]"
- End with: "- Your Care Team at ATLAS Health"

Return ONLY the message text, no explanation."""

        try:
            groq_client = Groq(api_key=GROQ_API_KEY)
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You write patient-friendly follow-up messages."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.5
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return (f"Hi {first_name}, this is your care team checking in. "
                   "How are you feeling? Please call us at [CLINIC_PHONE] "
                   "with any questions. - Your Care Team at ATLAS Health")