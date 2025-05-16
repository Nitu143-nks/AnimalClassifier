import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Ensure upload directory exists
os.makedirs('static/uploads', exist_ok=True)

# Load model
MODEL_PATH = 'model/animal_classifier_model.h5'
model = load_model(MODEL_PATH)

# Class names
class_names = ['Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin', 'Elephant',
               'Giraffe', 'Horse', 'Kangaroo', 'Lion', 'Panda', 'Tiger', 'Zebra']

# Route to homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join('static/uploads', filename)
        file.save(file_path)

        # Preprocess image
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Predict
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]

        return render_template('index.html', prediction=predicted_class, filename=filename)

    return 'Something went wrong'

if __name__ == '__main__':
    app.run(debug=True)
