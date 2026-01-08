# RL Policy Evaluation Server

FastAPI server for evaluating student RL policy submissions.

## Local Development

```bash
# Run the server
uv run uvicorn server.app:app --reload --port 8000

# Test submission
curl -X POST http://localhost:8000/submit \
  -F "student_id=test" \
  -F "policy_file=@server/sample_submission/policy.py" \
  -F "checkpoint_file=@server/sample_submission/checkpoint.pt"

# View leaderboard
curl http://localhost:8000/leaderboard/cartpole
```

## Deploy to Railway

1. Install Railway CLI:
   ```bash
   npm install -g @railway/cli
   ```

2. Login and initialize:
   ```bash
   railway login
   railway init
   ```

3. Create a service and link to it:
   ```bash
   railway add --service eval-server
   ```

4. Add a volume for persistent database storage:
   ```bash
   railway volume add
   # When prompted, set mount path to: /data
   ```

5. Set the DATA_DIR environment variable:
   ```bash
   railway variables --set "DATA_DIR=/data"
   ```

6. Deploy:
   ```bash
   railway up
   ```

7. Get your public URL:
   ```bash
   railway domain
   ```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/submit` | Submit policy for evaluation |
| GET | `/leaderboard/{env_name}` | View leaderboard |
| GET | `/submissions/{student_id}` | View student's submissions |
| GET | `/environments` | List available environments |
| GET | `/health` | Health check |
| GET | `/docs` | Interactive API documentation |

## Student Submission Format

Students submit two files:

### policy.py
```python
import torch
import torch.nn as nn

class MyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        # Define your network architecture
        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, obs):
        return self.net(obs).argmax(dim=-1)

def load_policy(checkpoint_path):
    """Required function - loads and returns the policy."""
    model = MyPolicy()
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()
    return model
```

### checkpoint.pt
PyTorch state dict saved with `torch.save(model.state_dict(), 'checkpoint.pt')`
