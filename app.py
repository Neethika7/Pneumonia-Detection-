import os
import numpy as np
import cv2
import joblib
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load pre-trained model and scaler
model = joblib.load('model/model.pkl')
scaler = joblib.load('model/scaler.pkl')

# Define labels (assuming binary classification: 0 for Normal, 1 for Pneumonia)
labels = {0: 'Normal', 1: 'Pneumonia'}

def preprocess_image(image_file):
    """Preprocess the uploaded image for prediction."""
    in_memory_file = image_file.read()  # Read the image data from the file
    nparr = np.frombuffer(in_memory_file, np.uint8)  # Convert to numpy array
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)  # Decode the image
    img = cv2.resize(img, (128, 128))  # Resize to 128x128 pixels
    img = img.flatten().reshape(1, -1)  # Flatten and reshape for the model
    return img

@app.route('/')
def index():
    """Render the index page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction."""
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Preprocess the image
    img = preprocess_image(file)
    
    # Scale the image data
    img = scaler.transform(img)

    # Predict the class
    prediction = model.predict(img)[0]
    result = labels[prediction]

    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
