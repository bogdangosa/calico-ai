# Use a verified uv image for building
FROM ghcr.io/astral-sh/uv:0.6.1-python3.12-bookworm AS builder

# Set the working directory
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy only dependency files first to leverage Docker cache
COPY pyproject.toml uv.lock ./

# Install dependencies into a virtual environment
RUN uv sync --frozen --no-install-project --no-dev

# --- Final Stage ---
# Full Debian Bookworm image
FROM python:3.12-bookworm

WORKDIR /app

# Copy the virtual environment from the builder
COPY --from=builder /app/.venv /app/.venv

# Ensure the virtual environment is used
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

# Copy the source code and configuration
COPY src/ ./src/
COPY config/ ./config/

# Create directories for outputs and models
RUN mkdir -p outputs/logs agent_models

# Expose the API port
EXPOSE 8000

# Default command: Start the API
# For training, this can be overridden in docker-compose or via CLI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
