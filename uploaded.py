import cv2
import customtkinter as cust
from PIL import Image, ImageTk
from customtkinter import CTkImage
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tkinter.filedialog import askopenfilename

cust.set_appearance_mode("dark")

class FaceRecognitionApp(cust.CTk):
    def __init__(self, title, width, height):
        super().__init__()

        self.title(title)
        self.geometry(f"{width}x{height}")

        self.model = load_model('face_recognition_model_trained_latest.h5')
        self.class_names = ['angelina jolie','brad pitt','denzel washington','hugh jackman','jennifer lawrence','johnny depp','kate winslet','leo dicaprio','megan fox','natalie portman','nicole kidman','robert downey jr','sandra bullock','scarlett johansson','tom cruise','tom hanks','will smith']

        self.control_frame = cust.CTkFrame(self)
        self.control_frame.pack(pady=20)

        self.upload_button = cust.CTkButton(self.control_frame, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10, padx=20)

        self.result_label = cust.CTkLabel(self.control_frame, text="", font=("Helvetica", 16))
        self.result_label.pack(pady=10, padx=20)

        self.image_frame = cust.CTkFrame(self)
        self.image_frame.pack(pady=20)

        self.image_label = cust.CTkLabel(self.image_frame, text="No Image Uploaded", font=("Helvetica", 16))
        self.image_label.pack()

        self.placeholder_img = Image.new('RGB', (640, 480), color='gray')
        self.ctk_placeholder_img = CTkImage(light_image=self.placeholder_img, dark_image=self.placeholder_img, size=(640, 480))
        self.image_label.configure(image=self.ctk_placeholder_img)

    def upload_image(self):
        file_path = askopenfilename()
        if file_path:
            img = cv2.imread(file_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            if len(faces) == 0:
                self.result_label.configure(text="No faces detected.")
                return

            for (x, y, w, h) in faces:
                crop_img = img[y:y+h, x:x+w]
                img_resized = cv2.resize(crop_img, (224, 224))
                img_preprocessed = preprocess_input(img_resized)
                img_expanded = np.expand_dims(img_preprocessed, axis=0)

                prediction = self.model.predict(img_expanded)
                classIndex = np.argmax(prediction, axis=1)
                probabilityValue = np.max(prediction)

                if probabilityValue > 0.5:
                    className = self.class_names[classIndex[0]]
                    label = f'{className} ({probabilityValue*100:.2f}%)'
                else:
                    label = "Unknown"

                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(cv2image)
            ctk_img = CTkImage(light_image=pil_img, dark_image=pil_img, size=(640, 480))
            self.image_label.configure(image=ctk_img, text="")
            self.result_label.configure(text="")

    def __del__(self):
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = FaceRecognitionApp("Face Recognition App", 1000, 800)
    app.mainloop()

# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array, load_img
# from tensorflow.keras.applications.efficientnet import preprocess_input

# # Load the trained model
# model = load_model('face_recognition_model_trained_latest.h5')
# class_names = ['angelina jolie','brad pitt','denzel washington','hugh jackman','jennifer lawrence','johnny depp','kate winslet','leo dicaprio','megan fox','natalie portman','nicole kidman','robert downey jr','sandra bullock','scarlett johansson','tom cruise','tom hanks','will smith']

# def preprocess_image(image):
#     img_resized = cv2.resize(image, (224, 224))
#     img_preprocessed = preprocess_input(img_resized)
#     return np.expand_dims(img_preprocessed, axis=0)

# # Path to the image used during training
# img_path = 'images/testing.jpg'
# img = load_img(img_path, target_size=(224, 224))
# img_array = img_to_array(img)
# img_preprocessed = preprocess_image(img_array)

# # Prediction
# prediction = model.predict(img_preprocessed)
# classIndex = np.argmax(prediction, axis=1)
# probabilityValue = np.max(prediction)

# print(f'Predicted class: {class_names[classIndex[0]]} with confidence {probabilityValue*100:.2f}%')

