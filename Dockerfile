# Start from an official RunPod image that has PyTorch and CUDA
FROM runpod/pytorch:1.0.2-cu1290-torch260-ubuntu2204
# This image provides torch==2.3.0 and torchaudio==2.3.0

# Set the working directory
WORKDIR /app

# Copy the requirements file and install packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy all your project code (the 'src' folder and 'runpod_handler.py')
COPY . .

# Set the default command to start the handler
CMD ["python", "runpod_handler.py"]