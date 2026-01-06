# RL Class 2025

Reinforcement Learning course project using [PufferLib](https://github.com/PufferAI/PufferLib) for training.

## Prerequisites

### Install uv

[uv](https://docs.astral.sh/uv/) is a fast Python package manager. Install it using one of the following methods:

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Homebrew (macOS):**
```bash
brew install uv
```

**pip:**
```bash
pip install uv
```

After installation, restart your terminal or run `source ~/.bashrc` (or equivalent) to ensure `uv` is available.

## Setup

1. Clone this repository:
```bash
git clone git@github.com:Emerge-Lab/RL-Class-25.git
cd RL-Class-25
```

2. Install dependencies:
```bash
uv sync
```

This will create a virtual environment and install all dependencies including PufferLib.

## Usage

Run commands using:
```bash
uv run python your_script.py
```

Or activate the virtual environment:
```bash
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows
```
