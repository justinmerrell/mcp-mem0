FROM python:3.12-slim

ARG PORT=8050

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock ./

# Create virtual environment and install dependencies
RUN python -m venv .venv
RUN uv pip install -e .

# Copy application code
COPY src/ ./src/
COPY public/ ./public/

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

EXPOSE ${PORT}

# Command to run the MCP server
CMD ["uv", "run", "src/main.py"]
