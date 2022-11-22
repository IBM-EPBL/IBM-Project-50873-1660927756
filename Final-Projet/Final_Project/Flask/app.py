#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, Response, render_template
import time
import pyttsx3
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# camera frames 
class Video(object):
	def __init__(self):
		self.video = cv2.VideoCapture(0)
		self.roi_start = (50, 150)
		self.roi_end = (250, 350)
		self.model = load_model("disaster1.h5") 
		self.index=['Cyclone','Earthquake','Flood','Wildfire']
		self.y = None
	def __del__(self):
		self.video.release()
	def get_frame(self):
		ret,frame = self.video.read()
		frame = cv2.resize(frame, (640, 480))
		copy = frame.copy()
		copy = copy[150:150+200,50:50+200]
		# Prediction Start
		cv2.imwrite('image.jpg',copy)
		copy_img = image.load_img('image.jpg', target_size=(64,64))
		x = image.img_to_array(copy_img)
		x = np.expand_dims(x, axis=0)
		pred = np.argmax(self.model.predict(x), axis=1)
		self.y = pred[0]
		cv2.putText(frame,'Prediction: '+str(self.index[self.y]),(170,50),cv2.FONT_HERSHEY_SIMPLEX,1,(16, 105, 137),2,cv2.LINE_AA)
		ret,jpg = cv2.imencode('.jpg', frame)
		var = str(self.index[self.y])
# 		text_speech = pyttsx3.init()
# 		text_speech.say(var)
# 		text_speech.runAndWait()
		return jpg.tobytes()


# flask 
app = Flask(__name__)
@app.route('/')
def index():
	return render_template('index.html')

def gen(camera):
	while True:
		frame = camera.get_frame()
		yield(b'--frame\r\n'
			b'Content-Type: image/jpeg\r\n\r\n' + frame +
			b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
	video = Video()
	return Response(gen(video), mimetype='multipart/x-mixed-replace; boundary = frame')

@app.route("/intro")
def intro():
	return render_template("intro.html")



if __name__ == '__main__':
	app.run()


# In[ ]:





# In[ ]:




