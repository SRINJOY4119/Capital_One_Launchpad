# Use Python 3.11 as base image (more stable than 3.13 for ML packages)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgfortran5 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Create a requirements file without Windows-specific packages
RUN grep -v "pywin32" requirements.txt > requirements_docker.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_docker.txt

# Install additional packages that might be missing
RUN pip install --no-cache-dir \
    fastapi==0.116.1 \
    uvicorn[standard]==0.30.6 \
    python-multipart==0.0.20 \
    python-dotenv==1.1.1

# Copy the entire application
COPY . .

# Create necessary directories
RUN mkdir -p /app/uploads /app/cache /app/models /app/logs

# Set permissions
RUN chmod -R 755 /app

# Expose the port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
