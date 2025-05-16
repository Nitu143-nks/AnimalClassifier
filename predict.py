import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Load saved model
model_path = "animal_classifier_model.h5"  # Ensure the model path is correct
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
model = load_model(model_path)

# Define class names (replace with your actual class names in the same order as training)
class_names = ['bear', 'bird', 'cat', 'cow', 'deer', 'dog', 'dolphin', 'elephant',
               'giraffe', 'horse', 'kangaroo', 'lion', 'panda', 'tiger', 'zebra']


# Specify the image path - assuming the image is inside the 'dataset' folder or its subfolders
img_path = "C:/Users/Nitesh k. sahoo/Desktop/AnimalClassifier/dataset/Bird/Bird_1_1.jpg"  # Replace with correct image path

# Check if the image exists
if not os.path.exists(img_path):
    raise FileNotFoundError(f"Image file not found at {img_path}")

# Load and preprocess the image
img = image.load_img(img_path, target_size=(224, 224))  # Adjust target size based on model requirements
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array /= 255.0  # Normalize

# Predict
predictions = model.predict(img_array)
predicted_class = class_names[np.argmax(predictions)]

print(f"Predicted class: {predicted_class}")
