FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATABASE_URL=postgresql://trader:trader_password@database:5432/trading_system \
    FLASK_ENV=production \
    PYTHONPATH=/app

# System dependencies (needed by numpy/pandas and visualization libraries)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        gcc \
        g++ \
        curl \
        pkg-config \
        libblas-dev \
        liblapack-dev \
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

# Install all dependencies from requirements.txt
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