import httpx
import asyncio


class FHIRClient:
    def __init__(self, base_url: str, token: str = None):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = 30.0

    def _get_headers(self) -> dict:
        headers = {"Content-Type": "application/fhir+json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def get_patient(self, patient_id: str) -> dict:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.get(
                    f"{self.base_url}/Patient/{patient_id}",
                    headers=self._get_headers()
                )
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as e:
                print(f"HTTP error fetching patient {patient_id}: {e.response.status_code}")
                raise
            except httpx.RequestError as e:
                print(f"Request error fetching patient {patient_id}: {e}")
                raise

    async def get_conditions(self, patient_id: str) -> list:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.get(
                    f"{self.base_url}/Condition",
                    params={"patient": patient_id},
                    headers=self._get_headers()
                )
                resp.raise_for_status()
                bundle = resp.json()
                return bundle.get("entry", [])
            except httpx.HTTPStatusError as e:
                print(f"HTTP error fetching conditions: {e.response.status_code}")
                raise
            except httpx.RequestError as e:
                print(f"Request error fetching conditions: {e}")
                raise

    async def get_medications(self, patient_id: str) -> list:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.get(
                    f"{self.base_url}/MedicationRequest",
                    params={"patient": patient_id},
                    headers=self._get_headers()
                )
                resp.raise_for_status()
                bundle = resp.json()
                return bundle.get("entry", [])
            except httpx.HTTPStatusError as e:
                print(f"HTTP error fetching medications: {e.response.status_code}")
                raise
            except httpx.RequestError as e:
                print(f"Request error fetching medications: {e}")
                raise

    async def get_observations(self, patient_id: str, category: str = "laboratory") -> list:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.get(
                    f"{self.base_url}/Observation",
                    params={"patient": patient_id, "category": category},
                    headers=self._get_headers()
                )
                resp.raise_for_status()
                bundle = resp.json()
                return bundle.get("entry", [])
            except httpx.HTTPStatusError as e:
                print(f"HTTP error fetching observations: {e.response.status_code}")
                raise
            except httpx.RequestError as e:
                print(f"Request error fetching observations: {e}")
                raise

    async def get_allergies(self, patient_id: str) -> list:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.get(
                    f"{self.base_url}/AllergyIntolerance",
                    params={"patient": patient_id},
                    headers=self._get_headers()
                )
                resp.raise_for_status()
                bundle = resp.json()
                return bundle.get("entry", [])
            except httpx.HTTPStatusError as e:
                print(f"HTTP error fetching allergies: {e.response.status_code}")
                raise
            except httpx.RequestError as e:
                print(f"Request error fetching allergies: {e}")
                raise

    async def get_full_context(self, patient_id: str) -> dict:
        patienttask = asyncio.create_task(self.get_patient(patient_id))
        conditionstask = asyncio.create_task(self.get_conditions(patient_id))
        medicationstask = asyncio.create_task(self.get_medications(patient_id))
        labstask = asyncio.create_task(self.get_observations(patient_id, "laboratory"))
        vitalstask = asyncio.create_task(self.get_observations(patient_id, "vital-signs"))
        allergiestask = asyncio.create_task(self.get_allergies(patient_id))

        results = await asyncio.gather(
            patienttask,
            conditionstask,
            medicationstask,
            labstask,
            vitalstask,
            allergiestask
        )

        return {
            "patient": results[0],
            "conditions": results[1],
            "medications": results[2],
            "labs": results[3],
            "vitals": results[4],
            "allergies": results[5]
        }