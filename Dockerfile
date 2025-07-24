# Multi-stage Docker build for Agent Trading System
FROM python:3.11-slim as builder

# Set environment variables for build optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install comprehensive build dependencies for all advanced frameworks
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
    libhdf5-dev \
    libffi-dev \
    libssl-dev \
    cmake \
    pkg-config \
    curl \
    git \
    wget \
    unzip \
    # Additional dependencies for advanced frameworks
    libboost-all-dev \
    swig \
    # For scientific computing
    libgsl-dev \
    # Python build dependencies
    python3-dev \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Copy requirements file
COPY requirements.txt ./

# Install core dependencies first (most stable packages)
RUN pip install --no-cache-dir \
    "numpy>=1.21.0,<2.0.0" \
    "flask>=2.3.0,<4.0.0" \
    "flask-cors>=4.0.0,<5.0.0" \
    "pandas>=1.5.0,<3.0.0" \
    "scipy>=1.9.0,<2.0.0" \
    "scikit-learn>=1.2.0,<2.0.0" \
    "matplotlib>=3.5.0,<4.0.0" \
    "seaborn>=0.12.0,<1.0.0" \
    "plotly>=5.15.0,<6.0.0" \
    "yfinance>=0.2.0,<1.0.0" \
    "requests>=2.25.0,<3.0.0" \
    "python-dotenv>=0.19.0,<2.0.0" \
    "sqlalchemy>=1.4.0,<3.0.0" \
    "psycopg2-binary>=2.9.0,<3.0.0" \
    "redis>=4.5.0,<5.0.0" \
    "celery>=5.3.0,<6.0.0" \
    "gunicorn>=21.0.0,<22.0.0" \
    "ta>=0.10.0,<1.0.0" \
    "ccxt>=4.0.0,<5.0.0" \
    "statsmodels>=0.14.0,<1.0.0" \
    "arch>=5.6.0,<7.0.0" \
    "fastapi>=0.100.0,<1.0.0" \
    "uvicorn>=0.20.0,<1.0.0" \
    "websockets>=11.0.0,<12.0.0" \
    "prometheus-client>=0.16.0,<1.0.0" \
    "structlog>=23.0.0,<24.0.0" \
    "pytest>=7.4.0,<8.0.0" \
    "pytest-asyncio>=0.21.0,<1.0.0"

# Install CrewAI and other essential packages that might have failed
RUN pip install --no-cache-dir \
    "crewai>=0.60.0,<0.64.0" \
    "crewai[tools]>=0.60.0,<0.64.0" \
    "langchain>=0.1.0,<1.0.0" \
    "langchain-community>=0.0.1,<1.0.0" \
    "langchain-openai>=0.0.1,<1.0.0" \
    "openai>=1.0.0,<2.0.0" \
    "quantlib>=1.31,<2.0" \
    "PyPortfolioOpt>=1.5.0,<2.0.0" \
    "tf-keras>=2.14.0,<3.0.0" \
    "tensorflow>=2.14.0,<3.0.0" \
    "torch>=2.0.0,<3.0.0" \
    "transformers>=4.30.0,<5.0.0"

# Install missing advanced ML and quant finance dependencies
RUN pip install --no-cache-dir \
    "hmmlearn>=0.3.0,<1.0.0" \
    "xgboost>=1.7.0,<3.0.0" \
    "lightgbm>=3.3.0,<5.0.0" \
    "stable-baselines3>=2.0.0,<3.0.0" \
    "pymc>=5.0.0,<6.0.0" \
    "arviz>=0.15.0,<1.0.0" \
    "pytensor>=2.8.0,<3.0.0" \
    "textblob>=0.17.0,<1.0.0" \
    "spacy>=3.6.0,<4.0.0" \
    "gymnasium>=0.28.0,<1.0.0"

# Install compatible ChromaDB version (for CrewAI embedchain)
RUN pip install --no-cache-dir \
    "chromadb>=0.4.0,<0.5.0" \
    "embedchain>=0.1.0,<1.0.0"

# Install any remaining requirements from the file, allowing some failures for optional packages
RUN pip install --no-cache-dir -r requirements.txt || echo "Some optional packages failed to install - core system ready"

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    APP_ENV=production \
    # Optimize Python performance
    MALLOC_TRIM_THRESHOLD_=100000 \
    MALLOC_MMAP_THRESHOLD_=100000 \
    # Advanced frameworks configuration
    OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    # Runtime libraries for compiled packages
    libgomp1 \
    libopenblas0-pthread \
    libgfortran5 \
    libhdf5-103-1 \
    libgsl27 \
    # Math libraries
    libblas3 \
    liblapack3 \
    # Additional runtime dependencies
    libffi8 \
    libssl3 \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user with proper home directory
RUN groupadd -r appuser && useradd -r -g appuser -m -d /home/appuser appuser

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Set working directory
WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY dashboard_api.py ./dashboard_api.py
COPY run.py ./run.py
COPY requirements.txt ./requirements.txt

# Copy environment files
COPY .env* ./
RUN if [ ! -f .env ] && [ -f .env.example ]; then cp .env.example .env; fi

# Create comprehensive directory structure for advanced features
RUN mkdir -p \
    /app/data \
    /app/logs \
    /app/models \
    /app/cache \
    /app/temp \
    /app/backups \
    /app/reports \
    /home/appuser/.cache \
    /home/appuser/.local \
    && chown -R appuser:appuser /app /home/appuser \
    && chmod 755 /home/appuser \
    && chmod -R 755 /app/data /app/logs /app/models /app/cache

# Switch to non-root user
USER appuser

# Set user-specific environment
ENV HOME=/home/appuser \
    USER=appuser

# Enhanced health check for advanced frameworks
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=5 \
    CMD python -c "import sys; sys.path.append('/app'); \
        from src.environment import validate_environment; \
        status = validate_environment(); \
        print(f'Health check: {status[\"status\"]}'); \
        exit(0 if status['status'] == 'healthy' else 1)" || exit 1

# Expose port
EXPOSE 8000

# Default command with enhanced logging
CMD ["python", "-u", "run.py"]
