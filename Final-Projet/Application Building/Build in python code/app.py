# Keras
from keras.models import load_model
from PIL import Image
import numpy as np
import io

import cv2  
import time

vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FPS, 1)
MODEL_PATH = 'Disaster_Classification_model.h5'
model = load_model(MODEL_PATH)


if vid.isOpened():
 ret, frame = vid.read()
 
 
 # continue to display window until 'q' is pressed
 while(True):
  try:
   image_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
   img = Image.open(io.BytesIO(image_bytes))
   img = img.resize((64,64))
   img = np.array(img)
   img = img / 255.0
   img = img.reshape(1,64,64,3)
   predictions = model.predict(img)
   pred = np.argmax(predictions, axis = 1)
   classes = ["Cyclone", "Earthquake", "Flood", "wildfire"]
   print(classes[pred[0]])
  except KeyboardInterrupt:
   print("---")

else:
    print("Cannot open camera")