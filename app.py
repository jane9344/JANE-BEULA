from flask import Flask, request, render_template
import joblib
import pandas as pd
import os
from extract_features import extract_features  # Ensure this module exists
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('final_model.pkl')

# Define a folder to save uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html', result="")

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    
    # If no file is selected
    if file.filename == '':
        return "No selected file", 400
    
    # Save the uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    
    try:
        # Extract features from the uploaded contract
        features = extract_features(file_path)
        
        # Reshape the features to be 2D (even if it's just one sample)
        feature_array = np.array(features).reshape(1, -1)  # Reshaping to (1, 6) assuming features are 6
        
        # Predict using the trained model
        prediction = model.predict(feature_array)
        
        # Show the result (0 for benign, 1 for malicious)
        result = "Benign" if prediction[0] == 0 else "Malicious"
        
    except Exception as e:
        result = f"Error during prediction: {str(e)}"
    
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
