from tkinter import filedialog
from typing import Self

import customtkinter as ctk
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from CNN import CNN


class EmotionRecognitionApp(ctk.CTk):
    def __init__(self: Self, model: CNN, device: str) -> None:
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self._setup_window()
        self._init_variables(model, device)
        self._create_layout()
        self._create_buttons()

    def _setup_window(self: Self) -> None:
        self.title("Emotion Recognition App")
        self.geometry("1400x700")
        self.resizable(False, False)

    def _init_variables(self: Self, model: CNN, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.image = None
        self.detected_faces = []
        self.image_size = (960, 540)

    def _create_layout(self: Self) -> None:
        # Main display area
        self.main_blob = ctk.CTkFrame(self, corner_radius=10)
        self.main_blob.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        # Button panel
        self.button_blob = ctk.CTkFrame(self, width=220, corner_radius=10)
        self.button_blob.pack(side="right", fill="both", padx=10, pady=10)

        # Label to display the image
        self.image_label = ctk.CTkLabel(
            self.main_blob, text="", width=self.image_size[0], height=self.image_size[1]
        )
        self.image_label.pack(fill="both", expand=True)

    def _create_buttons(self: Self) -> None:
        button_width = 150
        self.button_frame = ctk.CTkFrame(self.button_blob, fg_color="transparent")
        self.button_frame.pack(expand=True, padx=10, pady=10)

        # Select image button
        self.select_btn = ctk.CTkButton(
            self.button_frame,
            text="Select Image",
            command=self.load_image,
            width=button_width,
        )
        self.select_btn.pack(pady=15)

        # Detect face button
        self.detect_btn = ctk.CTkButton(
            self.button_frame,
            text="Detect Face",
            command=self.detect_face,
            width=button_width,
        )
        self.detect_btn.pack(pady=15)
        self.detect_btn.configure(state="disabled")

        # Predict emotion button
        self.predict_btn = ctk.CTkButton(
            self.button_frame,
            text="Predict Emotion",
            command=self.predict_emotion,
            width=button_width,
        )
        self.predict_btn.pack(pady=15)
        self.predict_btn.configure(state="disabled")

    def load_image(self: Self) -> None:
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = Image.open(file_path)
            self.image = self.image.resize(
                self.image_size
            )  # Resize image to fixed size
            self.detected_faces = []  # Reset any previously detected faces
            self.display_image(self.image)

            # Enable the face detection button and disable the prediction button
            self.detect_btn.configure(state="normal")
            self.predict_btn.configure(state="disabled")

    def display_image(self: Self, img: Image) -> None:
        ctk_img = ctk.CTkImage(
            img, size=self.image_size
        )  # Ensure image is displayed at the fixed size
        self.image_label.configure(image=ctk_img)
        self.image_label.image = ctk_img

    def detect_face(self: Self) -> None:
        if self.image:
            # Convert PIL image to OpenCV format
            img_cv = np.array(self.image.convert("RGB"))
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR

            # Load OpenCV pre-trained face detector
            face_cascade = cv2.CascadeClassifier(
                "./haarcascades/haarcascade_frontalface_default.xml"
            )
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            self.detected_faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=6
            )

            if len(self.detected_faces) > 0:
                for x, y, w, h in self.detected_faces:
                    cv2.rectangle(
                        img_cv, (x, y), (x + w, y + h), (205, 245, 95), 2
                    )  # Draw rectangle

                # Update the displayed image with rectangles
                self.image = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                self.display_image(self.image)

                # Enable the predict button after faces are detected
                self.predict_btn.configure(state="normal")
            else:
                # No faces detected, disable prediction
                self.predict_btn.configure(state="disabled")

    def predict_emotion(self: Self) -> None:
        if len(self.detected_faces) > 0:
            # Convert PIL image to OpenCV format
            img_cv = np.array(self.image.convert("RGB"))
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR

            for x, y, w, h in self.detected_faces:
                face_img = Image.fromarray(
                    cv2.cvtColor(img_cv[y : y + h, x : x + w], cv2.COLOR_BGR2RGB)
                )

                # Predict the emotion for the detected face
                emotion = self.get_emotion(face_img)

                # Draw emotion text above the rectangle
                img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)
                draw.text(
                    (x, y - 36),
                    emotion,
                    fill=(202, 61, 245),
                    font=ImageFont.truetype("arialbd.ttf", 28),
                )

                # Convert back to OpenCV format for display
                img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            # Update the displayed image with the emotions
            self.image = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            self.display_image(self.image)

    def get_emotion(self: Self, face_img: Image) -> str:
        transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((48, 48)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        face_tensor = transform(face_img)
        face_tensor = face_tensor.unsqueeze(0).to(self.device)

        emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
        with torch.no_grad():
            output = self.model(face_tensor)
            _, predicted = torch.max(output.data, 1)
            emotion_index = predicted.item()
        return emotions[emotion_index]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    model.load_state_dict(
        torch.load("./model/emotion_cnn.pth", map_location=device, weights_only=True)
    )
    model.eval()

    app = EmotionRecognitionApp(model=model, device=device)
    app.mainloop()


if __name__ == "__main__":
    main()
