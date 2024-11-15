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
    def __init__(self, weights, stream_url, output_dir="output_faces", shape_predictor_path="datasets/shape_predictor_68_face_landmarks.dat"):
        super().__init__()
        self.weights = weights
        self.stream_url = stream_url
        self.output_dir = output_dir
        self.shape_predictor_path = shape_predictor_path
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
        self.thread = threading.Thread(target=self.update_frame)
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
        """Extract and augment face from bounding box without adding annotations to saved images."""
        augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Resize((160, 160))
        ])

        x1, y1, x2, y2 = map(int, box)
        face = frame[y1:y2, x1:x2]

        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

        if smile_detected:
            filename = f"{name}_{nis}_smile.jpg"
            face_filename = os.path.join(self.output_dir, filename)
            face_pil.save(face_filename)
        else:
            for i in range(3):
                face_aug = augmentations(face_pil)
                face_aug = np.array(face_aug)
                face_aug = cv2.cvtColor(face_aug, cv2.COLOR_RGB2BGR)

                filename = f"{name}_{nis}_non_smile_aug_{i}.jpg"
                face_filename = os.path.join(self.output_dir, filename)
                cv2.imwrite(face_filename, face_aug)

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

    def update_frame(self):
        frame_counter = 0
        while True:
            success, frame = self.cap.read()
            if not success:
                print("Failed to capture frame from camera")
                continue

            # Resize frame for faster processing in live stream
            frame = cv2.resize(frame, (640, 480))
            frame_with_annotations = frame.copy()  # Copy frame for annotations

            frame_counter += 1
            if frame_counter % 1 != 0:
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

                    # Save highest confidence frames without annotations
                    if not self.capture_smile:
                        if confidence > self.highest_confidence_non_smile[0]:
                            self.highest_confidence_non_smile = (confidence, frame.copy(), result.xyxy[0])
                        self.non_smile_count += 1
                        if self.non_smile_count >= 50:
                            self.capture_smile = True
                            self.countdown_started = True  # Start the countdown

                    elif smile_detected:
                        if confidence > self.highest_confidence_smile[0]:
                            self.highest_confidence_smile = (confidence, frame.copy(), result.xyxy[0])
                        self.smile_count += 1
                        if self.smile_count >= 50:
                            # Save the highest confidence frames without annotations
                            if self.highest_confidence_non_smile[1] is not None:
                                self.extract_and_augment_faces(
                                    self.highest_confidence_non_smile[1],
                                    self.highest_confidence_non_smile[2],
                                    "name", "nis", self.highest_confidence_non_smile[0],
                                    smile_detected=False
                                )
                            if self.highest_confidence_smile[1] is not None:
                                self.extract_and_augment_faces(
                                    self.highest_confidence_smile[1],
                                    self.highest_confidence_smile[2],
                                    "name", "nis", self.highest_confidence_smile[0],
                                    smile_detected=True
                                )
                            return  # Stop processing after saving frames

                    # Draw bounding box and landmarks only for live stream view
                    cv2.rectangle(frame_with_annotations, (x1, y1), (x2, y2), (0, 255, 0) if not smile_detected else (255, 255, 0), 2)
                    label = f"Confidence: {confidence:.2f}"
                    cv2.putText(frame_with_annotations, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    self.draw_landmarks(frame_with_annotations, landmarks)

            # Display instructions for smiling
            if self.capture_smile and not self.countdown_started:
                cv2.putText(frame_with_annotations, "Silahkan Senyum...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Countdown for smiling
            if self.countdown_started:
                for i in range(3, 0, -1):
                    cv2.putText(frame_with_annotations, f"Senyum dalam {i}...", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    ret, buffer = cv2.imencode('.jpg', frame_with_annotations)
                    if ret:
                        with self.lock:
                            self.latest_frame = buffer.tobytes()
                    time.sleep(1)
                self.countdown_started = False
                self.smile_count = 0

            # Encode the frame with annotations for live streaming
            ret, buffer = cv2.imencode('.jpg', frame_with_annotations)
            if ret:
                with self.lock:
                    self.latest_frame = buffer.tobytes()

            time.sleep(0.02)

    def generate_frame(self, name, nis):
        while True:
            with self.lock:
                frame = self.latest_frame
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                time.sleep(0.05)

    def release(self):
        self.cap.release()
