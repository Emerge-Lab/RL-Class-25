# Use pre-built base image with all dependencies
# Build base with: docker build --platform linux/amd64 -t USERNAME/rl-eval-base -f Dockerfile.base .
# Push with: docker push USERNAME/rl-eval-base
FROM ghcr.io/emerge-lab/rl-eval-base:latest

WORKDIR /app

# Copy application code (this is the only thing that changes between deploys)
COPY server/ /app/server/

# Create data directory
RUN mkdir -p /data

ENV DATA_DIR=/data

EXPOSE 8000

# Use shell form so $PORT is expanded at runtime
CMD uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-8000}
