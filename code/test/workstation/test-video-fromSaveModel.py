import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from collections import deque

# Load the trained model saved in 'SavedModel' format
model = tf.saved_model.load('/Users/harishsenthilkumar/cv/fall_detection_model_saved')

# Open a video file or capture device
video_path = '/Users/harishsenthilkumar/Downloads/manfall.mp4'  # Change to your video path
output_dir = '/Users/harishsenthilkumar/Downloads/output_fall_detected'  # Output folder path

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)  # Create the directory if it doesn't exist

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties for output files
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Fall detection counter for naming files
fall_counter = 0
is_writing = False
out = None

# Buffer to store frames before and after fall detection
pre_fall_buffer = deque(maxlen=30)  # Store up to 30 frames before fall (adjust as needed)
post_fall_frames = 30  # Save 30 frames after fall detection ends (adjust as needed)
post_fall_counter = 0

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("End of video or error.")
        break

    # Resize the frame to match the input shape of the model
    frame_resized = cv2.resize(frame, (64, 64))
    
    # Convert the frame to an array and preprocess
    frame_array = image.img_to_array(frame_resized)
    frame_array = np.expand_dims(frame_array, axis=0)  # Add batch dimension
    frame_array /= 255.0  # Normalize to [0, 1]
    
    # Predict using the model (suppress logs)
    prediction = model.signatures['serving_default'](tf.convert_to_tensor(frame_array))['output_0']

    # Add frame to the pre-fall buffer
    pre_fall_buffer.append(frame)

    # If a fall is detected, start or continue writing the video
    if prediction.numpy()[0][0] <= 0.5:  # Assuming fall is detected when prediction <= 0.5
        if not is_writing:
            # Start a new video file for the detected fall
            fall_counter += 1
            output_video_path = os.path.join(output_dir, f'fall_detected_{fall_counter}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
            is_writing = True
            print(f"Started recording fall {fall_counter}: {output_video_path}")
            
            # Write the pre-fall buffer to the video
            while pre_fall_buffer:
                out.write(pre_fall_buffer.popleft())

        out.write(frame)  # Write current frame
    
    else:
        # If we were writing, continue to save frames for a few more frames after fall detection ends
        if is_writing:
            if post_fall_counter < post_fall_frames:
                out.write(frame)  # Save extra frames after fall
                post_fall_counter += 1
            else:
                # After saving extra frames, stop writing
                out.release()
                print(f"Stopped recording fall {fall_counter}")
                is_writing = False
                post_fall_counter = 0  # Reset counter for next fall

    # Optional: If you want to break the loop early with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer objects, and close all OpenCV windows
if out is not None:
    out.release()
cap.release()
cv2.destroyAllWindows()

print(f"Processed video and saved fall detections in {output_dir}")
