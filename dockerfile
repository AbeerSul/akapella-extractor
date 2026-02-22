FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements; install everything except torch/torchaudio (provided by base image)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir $(grep -vE '^(torch|torchaudio)' /app/requirements.txt | tr '\n' ' ')

# Ensure runpod is installed (in case requirements.txt is not picked up)
RUN pip install --no-cache-dir runpod

# Copy application
COPY . /app

RUN mkdir -p /app/output

EXPOSE 8000

CMD ["python", "app.py"]