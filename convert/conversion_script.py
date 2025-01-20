import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import numpy as np

# Define paths for model and output directories
SAVED_MODEL_DIR = '/workspace/model'  # Path to your saved TensorFlow model
OUTPUT_SAVED_MODEL_DIR = '/workspace/output/tftrt_saved_model'  # Output directory for TensorRT model

# Load the saved model
converter = trt.TrtGraphConverterV2(
    input_saved_model_dir=SAVED_MODEL_DIR,
    precision_mode=trt.TrtPrecisionMode.INT8,  # Change to FLOAT32 or FLOAT16 if necessary
    use_calibration=True  # Set to True if using INT8
)

# Calibration input function (example)
BATCH_SIZE = 32
NUM_CALIB_BATCHES = 10
def calibration_input_fn():
    for i in range(NUM_CALIB_BATCHES):
        # Update to match the expected input shape (e.g., 64x64 RGB images)
        x = np.random.rand(BATCH_SIZE, 64, 64, 3)  # Example shape; replace with actual calibration data
        yield [x.astype(np.float32)]

# Convert the model with calibration data
converter.convert(calibration_input_fn=calibration_input_fn)

print(f"********Script to convert 4d tensor")

# Input function for dynamic shapes profile generation
MAX_BATCH_SIZE = 128
def input_fn():
    # Update to match the expected input shape (e.g., 64x64 RGB images)
    x = np.random.rand(MAX_BATCH_SIZE, 64, 64, 3)  # Example shape; replace with actual data
    yield [x.astype(np.float32)]

# Build the engine
converter.build(input_fn=input_fn)

# Save the converted model
converter.save(output_saved_model_dir=OUTPUT_SAVED_MODEL_DIR)

# Summary of the conversion
converter.summary()
print(f"Model converted and saved to {OUTPUT_SAVED_MODEL_DIR}")

