"""
src/api/cibil_api.py
----------------------
FEATURE 4 — Mock CIBIL API.

`CibilBureauClient` is the abstraction the rest of the app depends on. Its
public method `fetch_score(pan, dob)` is called by the Streamlit app exactly
the way it would call a real credit bureau. Today `MockCibilProvider`
implements that contract with a deterministic, seeded-random simulation.
Swapping to a real bureau later means writing one new class
(e.g. `ExperianProvider`) that implements the same `.fetch_score()` method
and pointing `CIBIL_API_MODE=live` in .env — nothing else in the codebase
changes.

A thin FastAPI wrapper (`api_app`) is included so the same logic can also be
served as a real `POST /cibil` HTTP endpoint, matching the spec exactly.
Run it standalone with:
    uvicorn src.api.cibil_api:api_app --reload --port 8000
"""

from __future__ import annotations

import hashlib
import random
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CibilReport:
    cibil_score: int
    active_loans: int
    late_payments: int
    credit_utilization: int  # percentage
    credit_history_years: int

    def to_dict(self) -> dict:
        return asdict(self)


class BureauProvider(ABC):
    @abstractmethod
    def fetch_score(self, pan: str, dob: str) -> CibilReport:
        ...


class MockCibilProvider(BureauProvider):
    """
    Deterministic mock: the same (PAN, DOB) pair always returns the same
    report, so a demo or a test suite behaves consistently across reruns —
    while still looking like realistic, varied bureau data across different
    applicants.
    """

    def fetch_score(self, pan: str, dob: str) -> CibilReport:
        seed_material = f"{pan.strip().upper()}|{dob.strip()}".encode()
        seed = int(hashlib.sha256(seed_material).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)

        report = CibilReport(
            cibil_score=rng.randint(300, 900),
            active_loans=rng.randint(0, 5),
            late_payments=rng.randint(0, 6),
            credit_utilization=rng.randint(5, 95),
            credit_history_years=rng.randint(1, 20),
        )
        logger.info("Mock CIBIL report generated for PAN=%s: score=%s", pan, report.cibil_score)
        return report


class CibilBureauClient:
    """Facade used by the rest of the application."""

    def __init__(self, provider: BureauProvider | None = None) -> None:
        self._provider = provider or MockCibilProvider()

    def fetch_score(self, pan: str, dob: str) -> dict:
        if not pan or not dob:
            raise ValueError("PAN and date of birth are required to pull a CIBIL report.")
        report = self._provider.fetch_score(pan, dob)
        return report.to_dict()


# ---------------------------------------------------------------------------
# Optional standalone FastAPI service — matches the spec's literal
# "POST /cibil" endpoint contract, for teams that want CIBIL pulls behind
# a real network boundary instead of an in-process function call.
# ---------------------------------------------------------------------------
try:
    from fastapi import FastAPI
    from pydantic import BaseModel

    class CibilRequest(BaseModel):
        pan: str
        dob: str

    api_app = FastAPI(title="Mock CIBIL Bureau API")
    _client = CibilBureauClient()

    @api_app.post("/cibil")
    def cibil_endpoint(payload: CibilRequest) -> dict:
        return _client.fetch_score(payload.pan, payload.dob)

except ImportError:  # fastapi is optional — the Streamlit app never needs it
    api_app = None
