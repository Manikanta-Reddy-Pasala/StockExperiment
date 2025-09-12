FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DATABASE_URL=postgresql://trader:trader_password@database:5432/trading_system
ENV FLASK_ENV=production
ENV PYTHONPATH=/app

# Install system dependencies
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

# Copy requirements file first for better caching
COPY requirements.txt .

# Upgrade pip and install wheel
RUN pip install --upgrade pip setuptools wheel

# Install core dependencies first (these rarely change)
RUN pip install --no-cache-dir \
    numpy==1.23.5 \
    pandas==2.2.0

# Install ML dependencies (these change less frequently)
RUN pip install --no-cache-dir \
    scikit-learn==1.4.2 \
    xgboost==2.0.3 \
    optuna==3.6.1

# Install TensorFlow separately to avoid protobuf conflicts
# Using TensorFlow 2.11.0 which is compatible with protobuf 5.x
RUN pip install --no-cache-dir tensorflow==2.11.0

# Install remaining dependencies (these change most frequently)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create logs directory in the correct location and make it writable
RUN mkdir -p /app/src/data/logs && \
    chmod 777 /app/src/data/logs

# Expose port
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

# Run the application
CMD ["python", "run.py", "--multi-user"]