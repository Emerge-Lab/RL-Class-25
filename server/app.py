"""FastAPI server for policy submissions and leaderboard."""

import os
import shutil
import sqlite3
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from server.evaluate import evaluate_submission, EvalResult

app = FastAPI(title="RL Policy Evaluation Server")

# Use DATA_DIR env var for Railway volume, fallback to local directory
DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).parent))
DB_PATH = DATA_DIR / "leaderboard.db"


def init_db():
    """Initialize the SQLite database."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS submissions (
                id TEXT PRIMARY KEY,
                student_id TEXT NOT NULL,
                env_name TEXT NOT NULL,
                mean_reward REAL NOT NULL,
                std_reward REAL NOT NULL,
                mean_length REAL NOT NULL,
                episodes INTEGER NOT NULL,
                eval_time REAL NOT NULL,
                submitted_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_env_reward
            ON submissions(env_name, mean_reward DESC)
        """)
        conn.commit()


@contextmanager
def get_db():
    """Get a database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


class SubmissionResponse(BaseModel):
    submission_id: str
    student_id: str
    env_name: str
    mean_reward: float
    std_reward: float
    mean_length: float
    episodes: int
    eval_time: float
    rank: int


class LeaderboardEntry(BaseModel):
    rank: int
    student_id: str
    mean_reward: float
    std_reward: float
    submitted_at: str


@app.on_event("startup")
async def startup():
    init_db()


@app.post("/submit", response_model=SubmissionResponse)
async def submit_policy(
    student_id: str = Form(...),
    env_name: str = Form(default="cartpole"),
    policy_file: UploadFile = File(...),
    checkpoint_file: UploadFile = File(...),
    num_episodes: int = Form(default=100),
):
    """
    Submit a policy for evaluation.

    - student_id: Your unique identifier
    - env_name: Environment to evaluate on (default: cartpole)
    - policy_file: Python file defining load_policy(checkpoint_path)
    - checkpoint_file: PyTorch checkpoint (.pt file)
    - num_episodes: Number of episodes to run (default: 100)
    """
    # Validate file extensions
    if not policy_file.filename.endswith(".py"):
        raise HTTPException(400, "policy_file must be a .py file")
    if not checkpoint_file.filename.endswith(".pt"):
        raise HTTPException(400, "checkpoint_file must be a .pt file")

    # Create temporary directory for submission
    with tempfile.TemporaryDirectory() as tmpdir:
        submission_dir = Path(tmpdir)

        # Save uploaded files
        policy_path = submission_dir / "policy.py"
        checkpoint_path = submission_dir / "checkpoint.pt"

        with open(policy_path, "wb") as f:
            shutil.copyfileobj(policy_file.file, f)
        with open(checkpoint_path, "wb") as f:
            shutil.copyfileobj(checkpoint_file.file, f)

        # Run evaluation
        try:
            result = evaluate_submission(
                submission_dir,
                env_name=env_name,
                num_episodes=num_episodes,
                seed=42,
                timeout=60.0,
            )
        except Exception as e:
            raise HTTPException(400, f"Evaluation failed: {str(e)}")

    # Store in database
    submission_id = str(uuid.uuid4())[:8]
    submitted_at = datetime.utcnow().isoformat()

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO submissions
            (id, student_id, env_name, mean_reward, std_reward, mean_length, episodes, eval_time, submitted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                submission_id,
                student_id,
                env_name,
                result.mean_reward,
                result.std_reward,
                result.mean_length,
                result.episodes,
                result.eval_time,
                submitted_at,
            ),
        )
        conn.commit()

        # Get rank
        rank = conn.execute(
            """
            SELECT COUNT(*) + 1 FROM (
                SELECT student_id, MAX(mean_reward) as best
                FROM submissions
                WHERE env_name = ?
                GROUP BY student_id
                HAVING best > ?
            )
            """,
            (env_name, result.mean_reward),
        ).fetchone()[0]

    return SubmissionResponse(
        submission_id=submission_id,
        student_id=student_id,
        env_name=env_name,
        mean_reward=result.mean_reward,
        std_reward=result.std_reward,
        mean_length=result.mean_length,
        episodes=result.episodes,
        eval_time=result.eval_time,
        rank=rank,
    )


@app.get("/leaderboard/{env_name}", response_model=list[LeaderboardEntry])
async def get_leaderboard(env_name: str, limit: int = 50):
    """
    Get the leaderboard for an environment.

    Shows each student's best submission.
    """
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT student_id, MAX(mean_reward) as mean_reward, std_reward, submitted_at
            FROM submissions
            WHERE env_name = ?
            GROUP BY student_id
            ORDER BY mean_reward DESC
            LIMIT ?
            """,
            (env_name, limit),
        ).fetchall()

    return [
        LeaderboardEntry(
            rank=i + 1,
            student_id=row["student_id"],
            mean_reward=row["mean_reward"],
            std_reward=row["std_reward"],
            submitted_at=row["submitted_at"],
        )
        for i, row in enumerate(rows)
    ]


@app.get("/submissions/{student_id}")
async def get_student_submissions(student_id: str, env_name: str | None = None):
    """Get all submissions for a student."""
    with get_db() as conn:
        if env_name:
            rows = conn.execute(
                """
                SELECT * FROM submissions
                WHERE student_id = ? AND env_name = ?
                ORDER BY submitted_at DESC
                """,
                (student_id, env_name),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM submissions
                WHERE student_id = ?
                ORDER BY submitted_at DESC
                """,
                (student_id,),
            ).fetchall()

    return [dict(row) for row in rows]


@app.get("/environments")
async def list_environments():
    """List available environments for evaluation."""
    return {
        "environments": [
            {"name": "cartpole", "description": "Classic CartPole balancing task"},
        ]
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
