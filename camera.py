from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

def predict_mask(frame, face, mask):
	(h,w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300),(104.0, 177.0, 123.0))
	face.setInput(blob)
	detections = face.forward()

	faces = []
	locs = []
	preds = []

	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		if confidence > 90:
			box = directions[0,0,i,3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			(startX, startY) = (max(0, startX), max(0,startY))
			(endX, endY) = (min(w - 1, endX), min(h-1, endY))

			face = frame[startY:endY, startX:endX]
			if face.any():
				face = cv2.cvtColor(face, cv2.Color_BGR2RGB)
				face = cv2.resize(face,(224,224))
				face = img_to_array(face)
				face = preprocess_input(face)

				faces.append(face)
				locs.append((startX, startY, endX, endY))

	if len(faces)  >0:
		faces = np.array(faces, dtype="float32")
		preds = mask.predict(faces, batch_size=32)

	return (locs,preds)

print("Loading face detector!")

ptp = os.path.sep.join("face_detection", "deploy.prototxt")
wp = os.path.sep.join("face_detection", "face_detect.caffemodel")
fn = cv2.dnn.readNet(ptp, wp)

print("Loading face mask model!")

mask = load_model("mask_detector.model")


cam = cv2.VideoCapture(0)
time.sleep(2.0)
while True:
	ret_val, img  = cam.read()
	img = cv2.flip(img, 1)
	img = imutils.resize(frame, width=400)

	(locs, preds) = detect_and_predict_mask(img, fn, mask)

	for (box,pred) in zip(locs, preds):
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0,255,0) if label == "Mask" else (0,0,255)

		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		cv2.putText(frame,label,(startX,startY - 10),
				cv2.FONT_HERSHEY_SIMPLEX, .45, color, 2)
		cv2.rectangle(frame,(startX, startY), (endX, endY), color, 2)
	cv2.imshow('my webcam', img)
	if cv2.waitKey(1) == 27:
		break

cv2.destroyAllWindows()
