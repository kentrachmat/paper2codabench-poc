#!/usr/bin/env python3
"""
Minimal REST client for the Codabench API.

Usage:
    from codabench_client import CodabenchClient
    client = CodabenchClient(api_url, api_token)
    comp_id = client.upload_competition("bundle.zip")
"""
import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import Config


class CodabenchClient:
    """Thin wrapper around the Codabench REST API."""

    def __init__(self, api_url: str = None, api_token: str = None):
        self.api_url = (api_url or Config.CODABENCH_API_URL).rstrip("/")
        self.api_token = api_token or Config.CODABENCH_API_TOKEN
        if not self.api_token:
            raise ValueError("Codabench API token is required. Set CODABENCH_API_TOKEN in .env")

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_token}",
        })

    def _url(self, path: str) -> str:
        return f"{self.api_url}/{path.lstrip('/')}"

    def _check(self, resp: requests.Response) -> dict:
        if not resp.ok:
            raise RuntimeError(
                f"Codabench API error {resp.status_code}: {resp.text[:500]}"
            )
        if resp.headers.get("content-type", "").startswith("application/json"):
            return resp.json()
        return {"status": resp.status_code, "text": resp.text[:200]}

    # ------------------------------------------------------------------
    # Competition (bundle) management
    # ------------------------------------------------------------------

    def upload_competition(self, zip_path: str) -> dict:
        """
        Upload a competition bundle zip to Codabench.

        Args:
            zip_path: Path to the bundle zip file

        Returns:
            API response dict (typically includes competition id)
        """
        zip_path = Path(zip_path)
        if not zip_path.exists():
            raise FileNotFoundError(f"Zip not found: {zip_path}")

        with open(zip_path, "rb") as f:
            resp = self.session.post(
                self._url("/competitions/"),
                files={"zip_file": (zip_path.name, f, "application/zip")},
            )
        result = self._check(resp)
        print(f"Uploaded competition: {result}")
        return result

    def get_competition_status(self, competition_id: int) -> dict:
        """Check competition status."""
        resp = self.session.get(self._url(f"/competitions/{competition_id}/"))
        return self._check(resp)

    # ------------------------------------------------------------------
    # Submission management
    # ------------------------------------------------------------------

    def submit_solution(self, competition_id: int, solution_path: str, phase_id: int = None) -> dict:
        """
        Submit a solution file to a Codabench competition.

        Args:
            competition_id: Competition ID
            solution_path: Path to solution.py or zip
            phase_id: Optional phase ID (uses default phase if None)

        Returns:
            API response dict (typically includes submission id)
        """
        solution_path = Path(solution_path)
        if not solution_path.exists():
            raise FileNotFoundError(f"Solution not found: {solution_path}")

        data = {"competition_id": competition_id}
        if phase_id is not None:
            data["phase_id"] = phase_id

        with open(solution_path, "rb") as f:
            resp = self.session.post(
                self._url("/submissions/"),
                data=data,
                files={"file": (solution_path.name, f)},
            )
        result = self._check(resp)
        print(f"Submitted solution: {result}")
        return result

    def get_submission_status(self, submission_id: int) -> dict:
        """Check submission status and scores."""
        resp = self.session.get(self._url(f"/submissions/{submission_id}/"))
        return self._check(resp)


def main():
    """Quick smoke test — requires valid credentials."""
    Config.validate_codabench()
    client = CodabenchClient()
    print(f"Codabench client ready: {client.api_url}")
    print("Use client.upload_competition(zip_path) to upload a bundle.")


if __name__ == "__main__":
    main()
