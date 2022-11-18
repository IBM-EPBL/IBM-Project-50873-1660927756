# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# Keras
from keras.models import load_model
from PIL import Image
import numpy as np
import io

import cv2
vid = cv2.VideoCapture(0)

# Define a flask app
app = Flask(_name_)

# Model saved with Keras model.save()
MODEL_PATH = 'Disaster_Classification_model.h5'

# Load your trained model
model = load_model(MODEL_PATH)

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        img_bytes = f.read()
        img = Image.open(io.BytesIO(img_bytes))
        #img = uploaded_file.read()
        img = img.resize((64,64))
        img = np.array(img)
        img = img / 255.0
        img = img.reshape(1,64,64,3)

        predictions = model.predict(img)
        pred = np.argmax(predictions, axis = 1)
        classes = ["Cyclone", "Earthquake", "Flood", "wildfire"]
        print(classes[pred[0]])

        return classes[pred[0]]

    return None

if _name_ == '_main_':
    app.run(debug=True)