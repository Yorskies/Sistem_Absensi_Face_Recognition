import cv2
import torch
import numpy as np
import os
from ultralytics import YOLO
from flask import Response
from controllers.base_controller import BaseController
from torchvision import transforms
from PIL import Image
import threading
import time
import dlib

class YOLOController(BaseController):
    def __init__(self, weights, stream_url, output_dir="output_faces", 
                 shape_predictor_path="datasets/shape_predictor_68_face_landmarks.dat", 
                 name=None, nis=None):
        super().__init__()
        self.weights = weights
        self.stream_url = stream_url
        self.output_dir = output_dir
        self.shape_predictor_path = shape_predictor_path
        self.name = name  # Tambahkan variabel name
        self.nis = nis  # Tambahkan variabel nis
        self.model = self.get_model()
        self.cap = cv2.VideoCapture(self.stream_url)
        self.latest_frame = None
        self.lock = threading.Lock()

        # Initialize Dlib's shape predictor
        self.shape_predictor = dlib.shape_predictor(self.shape_predictor_path)

        os.makedirs(self.output_dir, exist_ok=True)

        # Variables to capture highest confidence frames
        self.highest_confidence_non_smile = (0, None, None)  # (confidence, frame, box)
        self.highest_confidence_smile = (0, None, None)

        # Flags to control capture
        self.capture_smile = False
        self.non_smile_count = 0
        self.smile_count = 0
        self.countdown_started = False

        # Start thread to continuously run YOLO detection
        if self.name and self.nis:
            self.thread = threading.Thread(target=self.update_frame, args=(self.name, self.nis))
            self.thread.daemon = True
            self.thread.start()

    def get_model(self):
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = YOLO(self.weights)
            model.to(device)
            self.log(f"Model loaded on {device}")
            return model
        except Exception as e:
            self.handle_error(f"Error loading model: {e}")

    def extract_and_augment_faces(self, frame, box, name, nis, confidence, smile_detected=False):
        """Extract, augment, and save faces with multiple augmentations."""
        # Tentukan direktori output berdasarkan name dan nis
        person_folder = os.path.join(self.output_dir, f"{name}_{nis}")
        os.makedirs(person_folder, exist_ok=True)  # Buat folder jika belum ada

        # Crop wajah berdasarkan bounding box
        x1, y1, x2, y2 = map(int, box)
        face = frame[y1:y2, x1:x2]
        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

        # Resize ke 160x160
        face_pil = face_pil.resize((160, 160))

        # Simpan gambar smile atau augmentasi
        if smile_detected:
            smile_filename = os.path.join(person_folder, "smile.jpg")
            face_pil.save(smile_filename)
            print(f"Saved smile image to: {smile_filename}")
        else:
            # Daftar augmentasi
            augmentations = [
                transforms.RandomHorizontalFlip(),  # Horizontal flip
                transforms.RandomRotation(15),  # Rotation
                transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Brightness and contrast
                transforms.RandomResizedCrop((160, 160), scale=(0.8, 1.0)),  # Resized crop
                transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2))  # Gaussian Blur
            ]

            for i, aug in enumerate(augmentations):
                # Terapkan augmentasi satu per satu
                augmented_face = aug(face_pil)
                augment_filename = os.path.join(person_folder, f"aug{i+1}.jpg")
                augmented_face.save(augment_filename)
                print(f"Saved augmented image to: {augment_filename}")

            # Tambahkan augmentasi dengan noise dan affine transform
            noisy_face = self.add_noise(np.array(face_pil))  # Tambahkan noise
            noisy_filename = os.path.join(person_folder, f"noisy.jpg")
            Image.fromarray((noisy_face * 255).astype(np.uint8)).save(noisy_filename)
            print(f"Saved noisy image to: {noisy_filename}")

            affine_face = self.affine_transform(face_pil)  # Tambahkan affine transform
            affine_filename = os.path.join(person_folder, f"affine.jpg")
            affine_face.save(affine_filename)
            print(f"Saved affine transformed image to: {affine_filename}")

    # Fungsi tambahan untuk augmentasi noise
    def add_noise(self, image):
        """Add random Gaussian noise to the image."""
        noise = np.random.normal(0, 0.05, image.shape)  # Gaussian noise
        noisy_image = np.clip(image / 255.0 + noise, 0, 1)  # Normalize and add noise
        return noisy_image

    # Fungsi tambahan untuk affine transform
    def affine_transform(self, image_pil):
        """Apply affine transformation to the image."""
        width, height = image_pil.size
        affine_params = (1, 0.2, -20, 0.1, 1, 15)  # Parameters for affine transform
        affine_image = image_pil.transform(
            (width, height), Image.AFFINE, affine_params, resample=Image.BICUBIC
        )
        return affine_image



    def draw_landmarks(self, frame, landmarks):
        """Draw facial landmarks on the frame."""
        for i in range(68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

    def is_smiling(self, landmarks):
        """Detect if the person is smiling based on mouth landmarks."""
        left_mouth = landmarks.part(48)
        right_mouth = landmarks.part(54)
        top_mouth = landmarks.part(51)
        bottom_mouth = landmarks.part(57)

        mouth_width = np.linalg.norm(np.array([left_mouth.x, left_mouth.y]) - np.array([right_mouth.x, right_mouth.y]))
        mouth_height = np.linalg.norm(np.array([top_mouth.x, top_mouth.y]) - np.array([bottom_mouth.x, bottom_mouth.y]))

        return mouth_height > 0 and (mouth_width / mouth_height) > 0.32

    def update_frame(self, name, nis):
        frame_counter = 0
        while True:
            success, frame = self.cap.read()
            if not success:
                print("Failed to capture frame from camera")
                break  # Hentikan loop jika kamera gagal membaca frame

            # Resize frame for faster processing
            frame = cv2.resize(frame, (640, 480))
            frame_with_annotations = frame.copy()

            frame_counter += 1
            if frame_counter % 1 != 0:  # Skip frames (adjust for performance)
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    with self.lock:
                        self.latest_frame = buffer.tobytes()
                time.sleep(0.05)
                continue

            results = self.model.predict(source=frame, verbose=False)

            for result in results[0].boxes:
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                confidence = result.conf.item()

                if confidence >= 0.55:
                    dlib_rect = dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2)
                    landmarks = self.shape_predictor(frame, dlib_rect)
                    smile_detected = self.is_smiling(landmarks)

                    if not self.capture_smile:
                        if confidence > self.highest_confidence_non_smile[0]:
                            self.highest_confidence_non_smile = (confidence, frame.copy(), result.xyxy[0])
                        self.non_smile_count += 1
                        if self.non_smile_count >= 50:  # Threshold to start capturing smiles
                            self.capture_smile = True
                            self.countdown_started = True

                    elif smile_detected:
                        if confidence > self.highest_confidence_smile[0]:
                            self.highest_confidence_smile = (confidence, frame.copy(), result.xyxy[0])
                        self.smile_count += 1
                        if self.smile_count >= 50:  # Threshold to stop capturing smiles
                            # Perform face extraction and augmentation
                            if self.highest_confidence_non_smile[1] is not None:
                                self.extract_and_augment_faces(
                                    self.highest_confidence_non_smile[1],
                                    self.highest_confidence_non_smile[2],
                                    name, nis, self.highest_confidence_non_smile[0],
                                    smile_detected=False
                                )
                            if self.highest_confidence_smile[1] is not None:
                                self.extract_and_augment_faces(
                                    self.highest_confidence_smile[1],
                                    self.highest_confidence_smile[2],
                                    name, nis, self.highest_confidence_smile[0],
                                    smile_detected=True
                                )
                            self.release()  # Matikan kamera setelah proses selesai
                            print("Extraction and augmentation completed. Stopping frame updates.")
                            return  # Hentikan proses loop setelah selesai

                    # Tambahkan anotasi ke frame
                    cv2.rectangle(frame_with_annotations, (x1, y1), (x2, y2), (0, 255, 0) if not smile_detected else (255, 255, 0), 2)
                    label = f"Confidence: {confidence:.2f}"
                    cv2.putText(frame_with_annotations, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    self.draw_landmarks(frame_with_annotations, landmarks)

            # Countdown untuk meminta senyum
            if self.capture_smile and not self.countdown_started:
                cv2.putText(frame_with_annotations, "Silahkan Senyum...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            if self.countdown_started:
                for i in range(3, 0, -1):
                    cv2.putText(frame_with_annotations, f"Senyum dalam {i}...", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    ret, buffer = cv2.imencode('.jpg', frame_with_annotations)
                    if ret:
                        with self.lock:
                            self.latest_frame = buffer.tobytes()
                    time.sleep(1)  # Delay untuk countdown
                self.countdown_started = False
                self.smile_count = 0

            # Encode frame terbaru untuk live streaming
            ret, buffer = cv2.imencode('.jpg', frame_with_annotations)
            if ret:
                with self.lock:
                    self.latest_frame = buffer.tobytes()

            time.sleep(0.02)  # Delay untuk menjaga frame rate stabil



    def generate_frame(self, name, nis):
        while True:
            # Pastikan frame terbaru tersedia sebelum mengirim
            if self.latest_frame is None:
                time.sleep(0.05)  # Tunggu sejenak jika belum ada frame
                continue

            # Kirim frame terbaru dalam format streaming MJPEG
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + self.latest_frame + b'\r\n')


    def release(self):
        """Release the camera and cleanup resources."""
        if self.cap.isOpened():
            self.cap.release()
            print("Camera released successfully.")
