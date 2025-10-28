# Start from RunPod image with CUDA 12.1 and Python 3.10
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set the working directory
WORKDIR /app

# Set the Hugging Face cache directory (points inside the volume)
ENV HF_HOME /runpod-volume/hf_cache

# Copy requirements file first
COPY requirements.txt .

# Install system dependencies needed by torchaudio
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Install python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code (src folder, handler script, etc.)
COPY . .
# Note: We don't need to copy test_input.json for RunPod

# Command to run the app
CMD ["python", "runpod_handler.py"]
