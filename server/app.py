"""FastAPI server for policy submissions and leaderboard."""

import os
import secrets
import shutil
import sqlite3
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Header, Depends, Request
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from server.evaluate import evaluate_submission, EvalResult

# Templates directory
TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=TEMPLATES_DIR)

app = FastAPI(title="RL Policy Evaluation Server")

# Use DATA_DIR env var for Railway volume, fallback to local directory
DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).parent))
DB_PATH = DATA_DIR / "leaderboard.db"

# Admin API key from environment variable
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "dev-admin-key")


def init_db():
    """Initialize the SQLite database."""
    with sqlite3.connect(DB_PATH) as conn:
        # Students table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                api_key TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_api_key ON students(api_key)
        """)

        # Submissions table
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


def generate_api_key() -> str:
    """Generate a secure API key."""
    return f"sk_{secrets.token_urlsafe(24)}"


def get_student_by_api_key(api_key: str) -> dict | None:
    """Look up student by API key."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM students WHERE api_key = ?",
            (api_key,)
        ).fetchone()
        return dict(row) if row else None


def verify_admin_key(x_admin_key: str = Header(...)) -> str:
    """Verify admin API key."""
    if x_admin_key != ADMIN_API_KEY:
        raise HTTPException(401, "Invalid admin key")
    return x_admin_key


def verify_student_key(x_api_key: str = Header(...)) -> dict:
    """Verify student API key and return student info."""
    student = get_student_by_api_key(x_api_key)
    if not student:
        raise HTTPException(401, "Invalid API key")
    return student


# Pydantic models
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


class StudentCreate(BaseModel):
    email: str


class StudentResponse(BaseModel):
    id: str
    email: str
    api_key: str
    created_at: str


@app.on_event("startup")
async def startup():
    init_db()


# ============ Admin Endpoints ============

@app.post("/admin/students", response_model=StudentResponse)
async def create_student(
    student: StudentCreate,
    _: str = Depends(verify_admin_key)
):
    """Create a new student account. Returns the API key (shown only once)."""
    student_id = str(uuid.uuid4())[:8]
    api_key = generate_api_key()
    created_at = datetime.utcnow().isoformat()

    with get_db() as conn:
        try:
            conn.execute(
                "INSERT INTO students (id, email, api_key, created_at) VALUES (?, ?, ?, ?)",
                (student_id, student.email, api_key, created_at)
            )
            conn.commit()
        except sqlite3.IntegrityError:
            raise HTTPException(400, f"Student with email {student.email} already exists")

    return StudentResponse(
        id=student_id,
        email=student.email,
        api_key=api_key,
        created_at=created_at
    )


@app.get("/admin/students")
async def list_students(_: str = Depends(verify_admin_key)):
    """List all students (without API keys for security)."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, email, created_at FROM students ORDER BY created_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


@app.delete("/admin/students/{student_id}")
async def delete_student(student_id: str, _: str = Depends(verify_admin_key)):
    """Delete a student account."""
    with get_db() as conn:
        result = conn.execute("DELETE FROM students WHERE id = ?", (student_id,))
        conn.commit()
        if result.rowcount == 0:
            raise HTTPException(404, "Student not found")
    return {"status": "deleted", "student_id": student_id}


@app.post("/admin/students/{student_id}/regenerate-key", response_model=StudentResponse)
async def regenerate_api_key(student_id: str, _: str = Depends(verify_admin_key)):
    """Regenerate API key for a student."""
    new_api_key = generate_api_key()

    with get_db() as conn:
        row = conn.execute("SELECT * FROM students WHERE id = ?", (student_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Student not found")

        conn.execute(
            "UPDATE students SET api_key = ? WHERE id = ?",
            (new_api_key, student_id)
        )
        conn.commit()

    return StudentResponse(
        id=row["id"],
        email=row["email"],
        api_key=new_api_key,
        created_at=row["created_at"]
    )


# ============ Student Endpoints ============

@app.post("/submit", response_model=SubmissionResponse)
async def submit_policy(
    student: dict = Depends(verify_student_key),
    env_name: str = Form(default="cartpole"),
    policy_file: UploadFile = File(...),
    checkpoint_file: UploadFile = File(...),
    num_episodes: int = Form(default=100),
):
    """
    Submit a policy for evaluation.

    Requires X-API-Key header for authentication.

    - env_name: Environment to evaluate on (default: cartpole)
    - policy_file: Python file defining load_policy(checkpoint_path)
    - checkpoint_file: PyTorch checkpoint (.pt file)
    - num_episodes: Number of episodes to run (default: 100)
    """
    student_id = student["email"]  # Use email as the display ID

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


# ============ Public Endpoints ============

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


@app.get("/api/my-submissions")
async def get_my_submissions(
    student: dict = Depends(verify_student_key),
    env_name: str | None = None
):
    """Get all submissions for the authenticated student (API endpoint)."""
    student_id = student["email"]

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


# ============ Web Pages ============

def get_environments_list():
    """Get list of available environments."""
    return [
        {"name": "cartpole", "description": "Classic CartPole balancing task"},
    ]


# Simple flash message support (stored in query params for simplicity)
def flash_context():
    """Return empty flash messages context (messages passed via template)."""
    return {"get_flashed_messages": lambda with_categories=False: []}


@app.get("/", response_class=HTMLResponse)
async def web_leaderboard(request: Request, env: str = "cartpole"):
    """Web page showing the leaderboard."""
    environments = get_environments_list()

    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT student_id, MAX(mean_reward) as mean_reward, std_reward, submitted_at
            FROM submissions
            WHERE env_name = ?
            GROUP BY student_id
            ORDER BY mean_reward DESC
            LIMIT 50
            """,
            (env,),
        ).fetchall()

    entries = [
        {
            "rank": i + 1,
            "student_id": row["student_id"],
            "mean_reward": row["mean_reward"],
            "std_reward": row["std_reward"],
            "submitted_at": row["submitted_at"],
        }
        for i, row in enumerate(rows)
    ]

    return templates.TemplateResponse(
        "leaderboard.html",
        {
            "request": request,
            "entries": entries,
            "environments": environments,
            "current_env": env,
            **flash_context(),
        },
    )


@app.get("/submit", response_class=HTMLResponse)
async def web_submit_form(request: Request):
    """Web page for submitting policies."""
    return templates.TemplateResponse(
        "submit.html",
        {
            "request": request,
            "environments": get_environments_list(),
            "error": None,
            "result": None,
            **flash_context(),
        },
    )


@app.post("/submit", response_class=HTMLResponse)
async def web_submit_policy(
    request: Request,
    api_key: str = Form(...),
    env_name: str = Form(default="cartpole"),
    policy_file: UploadFile = File(...),
    checkpoint_file: UploadFile = File(...),
    num_episodes: int = Form(default=100),
):
    """Handle web form submission."""
    # Verify API key
    student = get_student_by_api_key(api_key)
    if not student:
        return templates.TemplateResponse(
            "submit.html",
            {
                "request": request,
                "environments": get_environments_list(),
                "error": "Invalid API key",
                "result": None,
                **flash_context(),
            },
        )

    student_id = student["email"]

    # Validate file extensions
    if not policy_file.filename.endswith(".py"):
        return templates.TemplateResponse(
            "submit.html",
            {
                "request": request,
                "environments": get_environments_list(),
                "error": "Policy file must be a .py file",
                "result": None,
                **flash_context(),
            },
        )

    if not checkpoint_file.filename.endswith(".pt"):
        return templates.TemplateResponse(
            "submit.html",
            {
                "request": request,
                "environments": get_environments_list(),
                "error": "Checkpoint file must be a .pt file",
                "result": None,
                **flash_context(),
            },
        )

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
            return templates.TemplateResponse(
                "submit.html",
                {
                    "request": request,
                    "environments": get_environments_list(),
                    "error": f"Evaluation failed: {str(e)}",
                    "result": None,
                    **flash_context(),
                },
            )

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

    return templates.TemplateResponse(
        "submit.html",
        {
            "request": request,
            "environments": get_environments_list(),
            "error": None,
            "result": {
                "mean_reward": result.mean_reward,
                "std_reward": result.std_reward,
                "episodes": result.episodes,
                "rank": rank,
            },
            **flash_context(),
        },
    )


@app.get("/my-submissions", response_class=HTMLResponse)
async def web_my_submissions(request: Request, api_key: str | None = None):
    """Web page showing user's submissions."""
    if not api_key:
        return templates.TemplateResponse(
            "my_submissions.html",
            {
                "request": request,
                "api_key": None,
                "submissions": [],
                "student_email": None,
                "error": None,
                **flash_context(),
            },
        )

    student = get_student_by_api_key(api_key)
    if not student:
        return templates.TemplateResponse(
            "my_submissions.html",
            {
                "request": request,
                "api_key": None,
                "submissions": [],
                "student_email": None,
                "error": "Invalid API key",
                **flash_context(),
            },
        )

    student_id = student["email"]

    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT * FROM submissions
            WHERE student_id = ?
            ORDER BY submitted_at DESC
            """,
            (student_id,),
        ).fetchall()

    submissions = [dict(row) for row in rows]

    return templates.TemplateResponse(
        "my_submissions.html",
        {
            "request": request,
            "api_key": api_key,
            "submissions": submissions,
            "student_email": student_id,
            "error": None,
            **flash_context(),
        },
    )
