"""Tests for the student import script."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from server.import_students import import_students


class TestImportStudents:
    @pytest.fixture
    def mock_server(self):
        """Mock the requests.post to simulate server responses."""
        with patch("server.import_students.requests.post") as mock_post:
            yield mock_post

    def test_import_csv_with_header(self, mock_server):
        mock_server.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "email": "test@example.com",
                "api_key": "sk_test123",
                "id": "abc123",
                "created_at": "2024-01-01T00:00:00",
            },
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("email\n")
            f.write("test@example.com\n")
            f.flush()

            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as out:
                results = import_students(
                    Path(f.name),
                    "http://localhost:8000",
                    "admin-key",
                    Path(out.name),
                )

        assert len(results) == 1
        assert results[0]["email"] == "test@example.com"
        assert results[0]["api_key"] == "sk_test123"
        assert results[0]["status"] == "created"

    def test_import_csv_without_header(self, mock_server):
        mock_server.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "email": "plain@example.com",
                "api_key": "sk_plain123",
                "id": "def456",
                "created_at": "2024-01-01T00:00:00",
            },
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            # No header, just emails
            f.write("plain@example.com\n")
            f.write("another@example.com\n")
            f.flush()

            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as out:
                results = import_students(
                    Path(f.name),
                    "http://localhost:8000",
                    "admin-key",
                    Path(out.name),
                )

        assert len(results) == 2

    def test_import_handles_duplicates(self, mock_server):
        responses = [
            MagicMock(status_code=200, json=lambda: {
                "email": "new@example.com",
                "api_key": "sk_new",
                "id": "new123",
                "created_at": "2024-01-01",
            }),
            MagicMock(status_code=400, json=lambda: {"detail": "already exists"}),
        ]
        mock_server.side_effect = responses

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("email\n")
            f.write("new@example.com\n")
            f.write("existing@example.com\n")
            f.flush()

            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as out:
                results = import_students(
                    Path(f.name),
                    "http://localhost:8000",
                    "admin-key",
                    Path(out.name),
                )

        assert len(results) == 2
        assert results[0]["status"] == "created"
        assert results[1]["status"] == "already_exists"

    def test_import_handles_server_error(self, mock_server):
        mock_server.return_value = MagicMock(
            status_code=500,
            text="Internal Server Error",
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("email\n")
            f.write("test@example.com\n")
            f.flush()

            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as out:
                results = import_students(
                    Path(f.name),
                    "http://localhost:8000",
                    "admin-key",
                    Path(out.name),
                )

        assert len(results) == 1
        assert "error_500" in results[0]["status"]

    def test_import_handles_connection_error(self, mock_server):
        import requests
        mock_server.side_effect = requests.RequestException("Connection failed")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("email\n")
            f.write("test@example.com\n")
            f.flush()

            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as out:
                results = import_students(
                    Path(f.name),
                    "http://localhost:8000",
                    "admin-key",
                    Path(out.name),
                )

        assert len(results) == 1
        assert "error" in results[0]["status"]

    def test_import_empty_csv(self, mock_server):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("email\n")  # Header only
            f.flush()

            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as out:
                results = import_students(
                    Path(f.name),
                    "http://localhost:8000",
                    "admin-key",
                    Path(out.name),
                )

        assert len(results) == 0
        mock_server.assert_not_called()

    def test_import_skips_invalid_emails(self, mock_server):
        mock_server.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "email": "valid@example.com",
                "api_key": "sk_valid",
                "id": "valid123",
                "created_at": "2024-01-01",
            },
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("valid@example.com\n")
            f.write("not-an-email\n")  # No @ sign
            f.write("\n")  # Empty line
            f.flush()

            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as out:
                results = import_students(
                    Path(f.name),
                    "http://localhost:8000",
                    "admin-key",
                    Path(out.name),
                )

        # Should only import the valid email
        assert len(results) == 1
        assert results[0]["email"] == "valid@example.com"

    def test_output_csv_format(self, mock_server):
        mock_server.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "email": "test@example.com",
                "api_key": "sk_outputtest",
                "id": "out123",
                "created_at": "2024-01-01",
            },
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("email\n")
            f.write("test@example.com\n")
            f.flush()
            input_path = Path(f.name)

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as out:
            output_path = Path(out.name)

        import_students(input_path, "http://localhost:8000", "admin-key", output_path)

        # Read and verify output CSV
        import csv
        with open(output_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["email"] == "test@example.com"
        assert rows[0]["api_key"] == "sk_outputtest"
        assert rows[0]["student_id"] == "out123"
        assert rows[0]["status"] == "created"
