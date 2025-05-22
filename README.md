# 🐾 Animal Image Classifier

This project is a deep learning-powered web application built with Flask that classifies images of animals into one of 15 categories. It uses a pre-trained CNN model and provides a simple user interface for image upload and classification.

---

## 🧠 Project Objective

To build a system that can accurately classify images of animals using a convolutional neural network (CNN) model trained on a labeled dataset. The application provides an intuitive interface for users to upload an image and receive a prediction.

---

## 🚀 Features

📤 Upload animal images for classification  
🔍 Predicts the animal class among 15 categories:
- Cow, Deer, Dog, Dolphin, Elephant, Giraffe, Horse, Kangaroo, Lion, Panda, Tiger, Zebra

🧠 Utilizes a pre-trained deep learning model (`animal_classifier_model.h5`)  
📷 Real-time image processing and prediction  
🖥️ Simple and responsive UI (`index.html`)

---

## 🛠️ Tech Stack

| Layer       | Technology                |
|-------------|---------------------------|
| Frontend    | HTML5, CSS3 (Flask Template) |
| Backend     | Python, Flask             |
| DL Model    | Keras, TensorFlow         |
| Data        | Image dataset (15 classes)|
| Tools       | Git, GitHub, VS Code      |

---

## 📂 Folder Structure

ANIMALCLASSIFIER/
│
├── dataset/ # Dataset with 15 animal classes
│ ├── Cow/
│ ├── Deer/
│ ├── ...
│ └── Zebra/
│
├── model/
│ └── animal_classifier_model.h5 # Trained CNN model
│
├── static/ # Static assets (if any)
├── templates/
│ └── index.html # Web UI template
│
├── venv/ # Virtual environment
├── .gitignore # Files to ignore in Git
├── requirements.txt # Python dependencies
├── animal_classifier.py # Model training or testing script
├── predict.py # Image preprocessing and prediction logic
├── app.py # Flask application entry point
└── Procfile # For deployment on platforms like Heroku

yaml
Copy
Edit

---

## ⚙️ How to Run the Project Locally

**1. Clone the repository:**
```bash
git clone https://github.com/Nitu143-nks/AnimalImageClassifier.git
cd ANIMALCLASSIFIER
2. Create and activate a virtual environment (Windows):

bash
Copy
Edit
python -m venv venv
venv\Scripts\activate
3. Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
4. Run the Flask application:

bash
Copy
Edit
python app.py
5. Open in your browser:

cpp
Copy
Edit
http://127.0.0.1:5000/
🧠 Deep Learning Model
Developed using Keras with TensorFlow backend

CNN architecture trained on image dataset with 15 animal classes

Saved as animal_classifier_model.h5

Used for real-time predictions from uploaded images

📦 Dataset
Custom dataset with 15 labeled animal categories organized into folders (one per class) under dataset/. Used for training the CNN model.

🙌 Contributing
Feel free to contribute by improving UI, optimizing the model, or adding more animal classes. Start by opening an issue or submitting a pull request!

✨ Author
Nitesh Kumar Sahoo
📧 Email: sahoonitesh78@gmail.com
🎓 MCA Student | Full Stack Developer | ML Enthusiast
🌐 GitHub Profile
