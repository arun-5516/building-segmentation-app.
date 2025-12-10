# Dockerfile - Render-friendly image for CPU inference
FROM python:3.10-slim

# install OS deps (for opencv, shapely, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy requirements first for layer caching
COPY requirements.txt /app/requirements.txt

# upgrade pip
RUN pip install --upgrade pip

# Install PyTorch CPU wheel from official index (adjust if PyTorch version needed)
RUN pip install --index-url https://download.pytorch.org/whl/cpu/ torch torchvision --quiet

# install remaining Python deps
RUN pip install -r /app/requirements.txt

# copy everything else
COPY . /app

# Expose the port Render sets via $PORT
ENV PORT 10000
ENV PYTHONUNBUFFERED=1

# Use gunicorn as the production server binding to $PORT
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:${PORT}", "--workers", "1", "--threads", "4"]
