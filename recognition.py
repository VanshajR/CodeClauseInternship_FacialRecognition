import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk

# Load the trained model
model = load_model('face_recognition_model_trained_latest.h5')
class_names = ['angelina jolie','brad pitt','denzel washington','hugh jackman','jennifer lawrence','johnny depp','kate winslet','leo dicaprio','megan fox','natalie portman','nicole kidman','robert downey jr','sandra bullock','scarlett johansson','tom cruise','tom hanks','will smith']  # Update with your class names

# Initialize the face detector
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create the main application window
class FaceRecognitionApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)

        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()

        self.btn_snapshot = Button(window, text="Capture", width=50, command=self.capture)
        self.btn_snapshot.pack(anchor=tk.CENTER, expand=True)

        self.update()
        self.window.mainloop()

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            if len(faces) > 0:
                face_imgs = []
                for (x, y, w, h) in faces:
                    crop_img = frame[y:y+h, x:x+w]
                    img = cv2.resize(crop_img, (224, 224))  # EfficientNetB0 input size
                    img = preprocess_input(img)
                    face_imgs.append(img)

                face_imgs = np.array(face_imgs)
                predictions = model.predict(face_imgs)

                for (x, y, w, h), prediction in zip(faces, predictions):
                    classIndex = np.argmax(prediction)
                    probabilityValue = np.max(prediction)

                    if probabilityValue > 0.5:  # Confidence threshold
                        className = class_names[classIndex]
                        label = f'{className} ({probabilityValue*100:.2f}%)'
                    else:
                        label = "Unknown"

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(10, self.update)

    def capture(self):
        ret, frame = self.vid.read()
        if ret:
            cv2.imwrite("frame-capture.jpg", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

# Create the main window and start the app
FaceRecognitionApp(tk.Tk(), "Face Recognition App")
