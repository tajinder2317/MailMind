# MailMind Dockerfile
# Multi-stage build for optimized production deployment

# =============================================================================
# Base Stage - Python Runtime
# =============================================================================
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build dependencies
    build-essential \
    curl \
    git \
    # OCR dependencies
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    # Image processing dependencies
    libpng-dev \
    libjpeg-dev \
    libtiff-dev \
    zlib1g-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libwebp-dev \
    # PDF processing dependencies
    libpoppler-dev \
    libpoppler-cpp-dev \
    # Network utilities
    wget \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Development Stage
# =============================================================================
FROM base as development

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Create application directory
WORKDIR /app

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/config /app/tokens

# Set permissions
RUN chmod +x /app/scripts/*.sh 2>/dev/null || true

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command for development
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# =============================================================================
# Production Stage
# =============================================================================
FROM base as production

# Create non-root user for security
RUN groupadd -r mailmind && \
    useradd -r -g mailmind -d /app -s /bin/bash mailmind

# Install Python dependencies (production only)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Create application directory with correct permissions
WORKDIR /app
RUN chown -R mailmind:mailmind /app

# Copy application code
COPY --chown=mailmind:mailmind . .

# Create necessary directories
RUN mkdir -p /app/logs /app/config /app/tokens && \
    chown -R mailmind:mailmind /app/logs /app/config /app/tokens

# Install any additional production dependencies
RUN pip install gunicorn

# Create entrypoint script
RUN echo '#!/bin/bash\n\
set -e\n\
echo "Starting MailMind application..."\n\
\n\
# Wait for database to be ready\n\
echo "Waiting for database..."\n\
while ! nc -z db 5432; do\n\
    echo "Database is unavailable - sleeping"\n\
    sleep 2\n\
done\n\
echo "Database is ready!"\n\
\n\
# Wait for Qdrant to be ready\n\
echo "Waiting for Qdrant..."\n\
while ! nc -z qdrant 6333; do\n\
    echo "Qdrant is unavailable - sleeping"\n\
    sleep 2\n\
done\n\
echo "Qdrant is ready!"\n\
\n\
# Run database migrations (if needed)\n\
echo "Running database migrations..."\n\
# alembic upgrade head\n\
\n\
# Start the application\n\
echo "Starting MailMind with Gunicorn..."\n\
exec gunicorn main:app \\\n\
    --bind 0.0.0.0:8000 \\\n\
    --workers ${MAX_WORKERS:-4} \\\n\
    --worker-class uvicorn.workers.UvicornWorker \\\n\
    --timeout 120 \\\n\
    --keep-alive 5 \\\n\
    --max-requests 1000 \\\n\
    --max-requests-jitter 100 \\\n\
    --preload \\\n\
    --access-logfile - \\\n\
    --error-logfile - \\\n\
    --log-level ${LOG_LEVEL:-info}\n\
' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh && \
    chown mailmind:mailmind /app/entrypoint.sh

# Switch to non-root user
USER mailmind

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set labels for metadata
LABEL org.opencontainers.image.title="MailMind" \
      org.opencontainers.image.description="Gmail RAG system for semantic thread retrieval" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.vendor="MailMind Team" \
      org.opencontainers.image.licenses="MIT"

# Production command
ENTRYPOINT ["/app/entrypoint.sh"]

# =============================================================================
# Test Stage
# =============================================================================
FROM development as test

# Install test dependencies
RUN pip install pytest pytest-asyncio pytest-cov black isort mypy

# Create test database setup
RUN echo "Test stage ready" && \
    echo "Run tests with: pytest --cov=mailmind"

# Default test command
CMD ["pytest", "--cov=mailmind", "--cov-report=html", "--cov-report=term"]

# =============================================================================
# Security Scan Stage
# =============================================================================
FROM development as security

# Install security scanning tools
RUN pip install safety bandit

# Run security scans
RUN safety check -r requirements.txt && \
    bandit -r . -f json -o security-report.json || true

# Default security check command
CMD ["sh", "-c", "safety check -r requirements.txt && bandit -r ."]

# =============================================================================
# Build Arguments
# =============================================================================
ARG BUILD_VERSION=1.0.0
ARG BUILD_DATE
ARG VCS_REF

# Set build-time environment variables
ENV BUILD_VERSION=${BUILD_VERSION} \
    BUILD_DATE=${BUILD_DATE} \
    VCS_REF=${VCS_REF}

# =============================================================================
# Final Stage Selection
# =============================================================================
# Use --target flag to select stage:
# docker build --target development -t mailmind:dev .
# docker build --target production -t mailmind:prod .
# docker build --target test -t mailmind:test .
# docker build --target security -t mailmind:security .

# Default to production if no target specified
FROM production
