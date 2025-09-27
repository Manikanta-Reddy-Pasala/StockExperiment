FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATABASE_URL=postgresql://trader:trader_password@database:5432/trading_system \
    FLASK_ENV=production \
    PYTHONPATH=/app

# System dependencies (needed by numpy/pandas/scipy/TF, etc.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        gcc \
        g++ \
        curl \
        pkg-config \
        libhdf5-dev \
        libblas-dev \
        liblapack-dev \
        gfortran \
        libfreetype6-dev \
        libpng-dev \
        libjpeg-dev \
        libffi-dev \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip tooling
RUN pip install --upgrade pip setuptools wheel

# Pre-install core numeric stack with versions that satisfy TensorFlow
RUN pip install --no-cache-dir \
    numpy>=2.2.6 \
    pandas>=2.2.3

# Pre-install ML stack to avoid resolver backtracking later
RUN pip install --no-cache-dir \
    scikit-learn>=1.4.2 \
    xgboost>=2.0.3 \
    optuna>=3.6.1 \
    protobuf>=5.29.3 \
    tensorflow>=2.16.1

# Install remaining dependencies (app + tooling)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure logs directory exists and is writable
RUN mkdir -p /app/src/data/logs && chmod 777 /app/src/data/logs

# Expose port
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

# Run the application
CMD ["python", "run.py", "--multi-user"]