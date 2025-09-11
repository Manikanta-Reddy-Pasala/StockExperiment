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
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
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