import cv2
import os
import customtkinter as cust
from PIL import Image, ImageTk
from customtkinter import CTkImage

cust.set_appearance_mode("dark")

class FaceCaptureApp(cust.CTk):
    def __init__(self, title, width, height):
        super().__init__()

        self.title(title)
        self.geometry(f"{width}x{height}")

        self.cap = None
        self.facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.count = 0
        self.nameID = None
        self.path = None

        self.control_frame = cust.CTkFrame(self)
        self.control_frame.pack(pady=20)

        self.name_label = cust.CTkLabel(self.control_frame, text="Enter Your Name:", font=("Helvetica", 16))
        self.name_label.pack(pady=5, padx=20)

        self.name_entry = cust.CTkEntry(self.control_frame, width=200)
        self.name_entry.pack(pady=5, padx=20)

        self.capture_button = cust.CTkButton(self.control_frame, text="Start Capture", command=self.start_capture)
        self.capture_button.pack(pady=10, padx=20)

        self.stop_button = cust.CTkButton(self.control_frame, text="Stop Capture", command=self.stop_capture, state='disabled')
        self.stop_button.pack(pady=10, padx=20)

        self.video_frame = cust.CTkFrame(self)
        self.video_frame.pack(pady=20)

        self.video_label = cust.CTkLabel(self.video_frame, text="No Video Feed", font=("Helvetica", 16))
        self.video_label.pack()

        self.placeholder_img = Image.new('RGB', (640, 480), color='gray')
        self.ctk_placeholder_img = CTkImage(light_image=self.placeholder_img, dark_image=self.placeholder_img, size=(640, 480))
        self.video_label.configure(image=self.ctk_placeholder_img)

        self.running = False

    def start_capture(self):
        self.nameID = self.name_entry.get().strip().lower()
        if not self.nameID:
            print("Name cannot be empty")
            return

        self.path = 'images/' + self.nameID
        if os.path.exists(self.path):
            print("Name Already Taken")
            self.name_label.configure(text="Name Already Taken, Enter Again:")
            self.name_entry.delete(0, cust.END)
            return
        else:
            os.makedirs(self.path)

        self.cap = cv2.VideoCapture(0)
        self.count = 0
        self.running = True
        self.capture_button.configure(state='disabled')
        self.stop_button.configure(state='normal')
        self.geometry("700x700")
        self.update_frame()

    def stop_capture(self):
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.video_label.configure(image=self.ctk_placeholder_img, text="No Video Feed")
        self.capture_button.configure(state='normal')
        self.stop_button.configure(state='disabled')
        self.geometry("400x400")

    def update_frame(self):
        if self.running:
            ret, frame = self.cap.read()
            if ret:
                faces = self.facedetect.detectMultiScale(frame, 1.3, 5)
                for x, y, w, h in faces:
                    self.count += 1
                    name = os.path.join(self.path, f'{self.count}.jpg')
                    print(f"Creating Images.........{name}")
                    cv2.imwrite(name, frame[y:y+h, x:x+w])
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                ctk_img = CTkImage(light_image=img, dark_image=img, size=(640, 480))
                self.video_label.configure(image=ctk_img, text="")

                if self.count > 500:
                    self.stop_capture()

        self.after(10, self.update_frame)

    def __del__(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = FaceCaptureApp("Face Capture App", 400, 400)
    app.mainloop()
