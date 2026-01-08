"""Tests for the FastAPI evaluation server."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

# Set test environment before importing app
os.environ["ADMIN_API_KEY"] = "test-admin-key"


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("server.app.DATA_DIR", Path(tmpdir)):
            with patch("server.app.DB_PATH", Path(tmpdir) / "leaderboard.db"):
                # Re-import to pick up patched paths
                from server import app as app_module
                app_module.DB_PATH = Path(tmpdir) / "leaderboard.db"
                app_module.init_db()
                yield Path(tmpdir)


@pytest.fixture
def client(temp_db):
    """Create a test client with temporary database."""
    from server.app import app
    with TestClient(app) as client:
        yield client


@pytest.fixture
def admin_headers():
    """Headers for admin authentication."""
    return {"X-Admin-Key": "test-admin-key"}


@pytest.fixture
def student_api_key(client, admin_headers):
    """Create a test student and return their API key."""
    response = client.post(
        "/admin/students",
        json={"email": "test@example.com"},
        headers=admin_headers,
    )
    return response.json()["api_key"]


@pytest.fixture
def student_headers(student_api_key):
    """Headers for student authentication."""
    return {"X-API-Key": student_api_key}


class TestHealthEndpoint:
    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestEnvironmentsEndpoint:
    def test_list_environments(self, client):
        response = client.get("/environments")
        assert response.status_code == 200
        data = response.json()
        assert "environments" in data
        assert len(data["environments"]) > 0
        assert data["environments"][0]["name"] == "cartpole"


class TestAdminEndpoints:
    def test_create_student(self, client, admin_headers):
        response = client.post(
            "/admin/students",
            json={"email": "newstudent@example.com"},
            headers=admin_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "newstudent@example.com"
        assert data["api_key"].startswith("sk_")
        assert "id" in data
        assert "created_at" in data

    def test_create_duplicate_student(self, client, admin_headers):
        # Create first student
        client.post(
            "/admin/students",
            json={"email": "duplicate@example.com"},
            headers=admin_headers,
        )
        # Try to create duplicate
        response = client.post(
            "/admin/students",
            json={"email": "duplicate@example.com"},
            headers=admin_headers,
        )
        assert response.status_code == 400
        assert "already exists" in response.json()["detail"]

    def test_create_student_without_admin_key(self, client):
        response = client.post(
            "/admin/students",
            json={"email": "test@example.com"},
        )
        assert response.status_code == 422  # Missing header

    def test_create_student_with_wrong_admin_key(self, client):
        response = client.post(
            "/admin/students",
            json={"email": "test@example.com"},
            headers={"X-Admin-Key": "wrong-key"},
        )
        assert response.status_code == 401

    def test_list_students(self, client, admin_headers):
        # Create some students
        client.post(
            "/admin/students",
            json={"email": "student1@example.com"},
            headers=admin_headers,
        )
        client.post(
            "/admin/students",
            json={"email": "student2@example.com"},
            headers=admin_headers,
        )

        response = client.get("/admin/students", headers=admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 2
        # API keys should not be included
        for student in data:
            assert "api_key" not in student
            assert "email" in student
            assert "id" in student

    def test_delete_student(self, client, admin_headers):
        # Create a student
        create_response = client.post(
            "/admin/students",
            json={"email": "todelete@example.com"},
            headers=admin_headers,
        )
        student_id = create_response.json()["id"]

        # Delete the student
        response = client.delete(
            f"/admin/students/{student_id}",
            headers=admin_headers,
        )
        assert response.status_code == 200
        assert response.json()["status"] == "deleted"

        # Verify student is gone
        list_response = client.get("/admin/students", headers=admin_headers)
        emails = [s["email"] for s in list_response.json()]
        assert "todelete@example.com" not in emails

    def test_delete_nonexistent_student(self, client, admin_headers):
        response = client.delete(
            "/admin/students/nonexistent",
            headers=admin_headers,
        )
        assert response.status_code == 404

    def test_regenerate_api_key(self, client, admin_headers):
        # Create a student
        create_response = client.post(
            "/admin/students",
            json={"email": "regenerate@example.com"},
            headers=admin_headers,
        )
        student_id = create_response.json()["id"]
        old_api_key = create_response.json()["api_key"]

        # Regenerate key
        response = client.post(
            f"/admin/students/{student_id}/regenerate-key",
            headers=admin_headers,
        )
        assert response.status_code == 200
        new_api_key = response.json()["api_key"]
        assert new_api_key != old_api_key
        assert new_api_key.startswith("sk_")

    def test_regenerate_key_nonexistent_student(self, client, admin_headers):
        response = client.post(
            "/admin/students/nonexistent/regenerate-key",
            headers=admin_headers,
        )
        assert response.status_code == 404


class TestAuthentication:
    def test_submit_without_api_key(self, client):
        response = client.post("/submit")
        assert response.status_code == 422  # Missing header

    def test_submit_with_invalid_api_key(self, client):
        response = client.post(
            "/submit",
            headers={"X-API-Key": "invalid-key"},
        )
        assert response.status_code == 401

    def test_my_submissions_without_api_key(self, client):
        response = client.get("/my-submissions")
        assert response.status_code == 422

    def test_my_submissions_with_invalid_api_key(self, client):
        response = client.get(
            "/my-submissions",
            headers={"X-API-Key": "invalid-key"},
        )
        assert response.status_code == 401


class TestSubmissionEndpoint:
    @pytest.fixture
    def sample_policy_file(self):
        """Create a temporary policy file."""
        content = '''
import torch
import torch.nn as nn

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(4, 2)

    def forward(self, x):
        return self.net(x).argmax(dim=-1)

def load_policy(checkpoint_path):
    return Policy()
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            yield f.name
        os.unlink(f.name)

    @pytest.fixture
    def sample_checkpoint_file(self):
        """Create a temporary checkpoint file."""
        import torch
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({}, f.name)
            yield f.name
        os.unlink(f.name)

    def test_submit_invalid_policy_extension(self, client, student_headers):
        with tempfile.NamedTemporaryFile(suffix=".txt") as policy:
            with tempfile.NamedTemporaryFile(suffix=".pt") as checkpoint:
                response = client.post(
                    "/submit",
                    headers=student_headers,
                    files={
                        "policy_file": ("policy.txt", policy, "text/plain"),
                        "checkpoint_file": ("checkpoint.pt", checkpoint, "application/octet-stream"),
                    },
                )
        assert response.status_code == 400
        assert ".py" in response.json()["detail"]

    def test_submit_invalid_checkpoint_extension(self, client, student_headers):
        with tempfile.NamedTemporaryFile(suffix=".py") as policy:
            with tempfile.NamedTemporaryFile(suffix=".txt") as checkpoint:
                response = client.post(
                    "/submit",
                    headers=student_headers,
                    files={
                        "policy_file": ("policy.py", policy, "text/plain"),
                        "checkpoint_file": ("checkpoint.txt", checkpoint, "text/plain"),
                    },
                )
        assert response.status_code == 400
        assert ".pt" in response.json()["detail"]

    def test_submit_valid_policy(
        self, client, student_headers, sample_policy_file, sample_checkpoint_file
    ):
        with patch("server.app.evaluate_submission") as mock_eval:
            from server.evaluate import EvalResult
            mock_eval.return_value = EvalResult(
                mean_reward=100.0,
                std_reward=10.0,
                mean_length=50.0,
                episodes=100,
                eval_time=1.5,
            )

            with open(sample_policy_file, "rb") as policy:
                with open(sample_checkpoint_file, "rb") as checkpoint:
                    response = client.post(
                        "/submit",
                        headers=student_headers,
                        data={"env_name": "cartpole", "num_episodes": "10"},
                        files={
                            "policy_file": ("policy.py", policy, "text/plain"),
                            "checkpoint_file": ("checkpoint.pt", checkpoint, "application/octet-stream"),
                        },
                    )

            assert response.status_code == 200
            data = response.json()
            assert data["mean_reward"] == 100.0
            assert data["std_reward"] == 10.0
            assert data["episodes"] == 100
            assert "submission_id" in data
            assert "rank" in data


class TestLeaderboardEndpoint:
    def test_empty_leaderboard(self, client):
        response = client.get("/leaderboard/cartpole")
        assert response.status_code == 200
        assert response.json() == []

    def test_leaderboard_with_submissions(self, client, admin_headers):
        # Create students and mock submissions
        from server.app import get_db
        import uuid
        from datetime import datetime

        # Create test submissions directly in DB
        with get_db() as conn:
            for i, (email, reward) in enumerate([
                ("top@example.com", 500.0),
                ("mid@example.com", 300.0),
                ("low@example.com", 100.0),
            ]):
                conn.execute(
                    """
                    INSERT INTO submissions
                    (id, student_id, env_name, mean_reward, std_reward, mean_length, episodes, eval_time, submitted_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4())[:8],
                        email,
                        "cartpole",
                        reward,
                        10.0,
                        50.0,
                        100,
                        1.0,
                        datetime.utcnow().isoformat(),
                    ),
                )
            conn.commit()

        response = client.get("/leaderboard/cartpole")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3
        assert data[0]["rank"] == 1
        assert data[0]["student_id"] == "top@example.com"
        assert data[0]["mean_reward"] == 500.0
        assert data[1]["rank"] == 2
        assert data[2]["rank"] == 3

    def test_leaderboard_limit(self, client):
        response = client.get("/leaderboard/cartpole?limit=2")
        assert response.status_code == 200


class TestMySubmissionsEndpoint:
    def test_my_submissions_empty(self, client, student_headers):
        response = client.get("/my-submissions", headers=student_headers)
        assert response.status_code == 200
        assert response.json() == []

    def test_my_submissions_with_data(self, client, student_headers):
        from server.app import get_db
        import uuid
        from datetime import datetime

        # Add submission for this student
        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO submissions
                (id, student_id, env_name, mean_reward, std_reward, mean_length, episodes, eval_time, submitted_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4())[:8],
                    "test@example.com",
                    "cartpole",
                    200.0,
                    15.0,
                    60.0,
                    100,
                    2.0,
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()

        response = client.get("/my-submissions", headers=student_headers)
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["student_id"] == "test@example.com"
        assert data[0]["mean_reward"] == 200.0

    def test_my_submissions_filter_by_env(self, client, student_headers):
        response = client.get(
            "/my-submissions?env_name=cartpole",
            headers=student_headers,
        )
        assert response.status_code == 200
