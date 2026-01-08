# Use pre-built base image with all dependencies
# Build base with: docker build --platform linux/amd64 -t USERNAME/rl-eval-base -f Dockerfile.base .
# Push with: docker push USERNAME/rl-eval-base
FROM ghcr.io/eugenevinitsky/rl-eval-base:latest

WORKDIR /app

# Copy application code (this is the only thing that changes between deploys)
COPY server/ /app/server/

# Create data directory
RUN mkdir -p /data

ENV DATA_DIR=/data
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
