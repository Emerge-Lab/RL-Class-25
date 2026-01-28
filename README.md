# RL Class 2025

Reinforcement Learning course project using [PufferLib](https://github.com/PufferAI/PufferLib) for training.

## Prerequisites

### Install uv

[uv](https://docs.astral.sh/uv/) is a fast Python package manager. Install it using one of the following methods:

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
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

### WSL (Windows Subsystem for Linux)

If you're on Windows, we recommend using WSL for a better development experience.

**1. Install WSL (if not already installed):**

Open PowerShell as Administrator and run:
```powershell
wsl --install
```

This installs WSL 2 with Ubuntu by default. Restart your computer when prompted.

**2. Set up your WSL environment:**

Open the Ubuntu terminal (search for "Ubuntu" in the Start menu) and run:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git curl
```

**3. Install uv inside WSL:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

**4. (Optional) GPU Support for PyTorch:**

If you have an NVIDIA GPU and want to use CUDA:
- Install the latest NVIDIA drivers on Windows (not inside WSL)
- WSL 2 automatically provides GPU access—no need to install CUDA inside WSL separately
- Verify GPU access: `nvidia-smi`

**5. Clone and work from WSL:**

Clone the repository to your WSL filesystem (not `/mnt/c/...`) for better performance:
```bash
cd ~
git clone git@github.com:Emerge-Lab/RL-Class-25.git
cd RL-Class-25
uv sync
```

**Tips:**
- Access WSL files from Windows Explorer: `\\wsl$\Ubuntu\home\<username>`
- Use VS Code with the "Remote - WSL" extension for seamless editing

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
