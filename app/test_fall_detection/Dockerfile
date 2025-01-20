# Use NVIDIA's TensorFlow image with integrated GPU support
FROM nvcr.io/nvidia/tensorflow:24.10-tf2-py3-igpu

# Set the working directory
WORKDIR /workspace

# Install necessary packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && pip3 install --upgrade pip \
    && pip3 install opencv-python-headless

# Copy the conversion script to the container
COPY test-video-fromSaveModel.py /workspace/test-video-fromSaveModel.py

# Command to run the conversion script
ENTRYPOINT ["python3", "/workspace/test-video-fromSaveModel.py"]

