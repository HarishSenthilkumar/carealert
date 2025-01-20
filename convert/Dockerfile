# Use NVIDIA's TensorFlow image with integrated GPU support
FROM nvcr.io/nvidia/tensorflow:24.10-tf2-py3-igpu

# Set the working directory
WORKDIR /workspace

# Install necessary packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && pip3 install --upgrade pip

# Copy the conversion script to the container
COPY conversion_script.py /workspace/conversion_script.py

# Command to run the conversion script
ENTRYPOINT ["python3", "/workspace/conversion_script.py"]

