import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define the image data generator for training data with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,            # Normalize pixel values to [0, 1]
    shear_range=0.2,           # Randomly apply shearing transformations
    zoom_range=0.2,            # Randomly zoom into images
    horizontal_flip=True       # Randomly flip images horizontally
)

# Load the training data
train_generator = train_datagen.flow_from_directory(
    '/Users/harishsenthilkumar/Downloads/cv_trainning_data',  # Path to the training data directory
    target_size=(64, 64),         # Resize images to this size
    batch_size=32,                # Number of images to return in each batch
    class_mode='binary'           # Binary classification (fall or no fall)
)

# Define the image data generator for validation data
val_datagen = ImageDataGenerator(rescale=1./255)  # Normalize pixel values

# Load the validation data
validation_generator = val_datagen.flow_from_directory(
    '/Users/harishsenthilkumar/Downloads/cv_testing_data',  # Path to the validation data directory
    target_size=(64, 64),            # Resize images to this size
    batch_size=32,                   # Number of images to return in each batch
    class_mode='binary'              # Binary classification (fall or no fall)
)

# Define the model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),  # Additional convolutional layer
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),  # Increase dense layer size
    Dropout(0.5),                   # Dropout to prevent overfitting
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=20,  # Increased number of epochs for better learning
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Use the export method for SavedModel export in Keras 3
model.export('/Users/harishsenthilkumar/cv/fall_detection_model_saved')

# Evaluate the model on the validation data
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f'Validation Loss: {val_loss}')
print(f'Validation Accuracy: {val_accuracy}')
print(train_generator.class_indices)