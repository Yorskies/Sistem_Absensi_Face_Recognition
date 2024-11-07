import threading
from ultralytics import YOLO
import torch
import cv2
from controllers.base_controller import BaseController

class YOLOController(BaseController):
    def __init__(self, weights, stream_url):
        super().__init__()
        self.weights = weights
        self.stream_url = stream_url
        self.model = self.get_model()
        self.current_frame = None
        self.ret = False
        self.lock = threading.Lock()
        self.cap = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)

    def get_model(self):
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            device = 'mps' if torch.backends.mps.is_available() else device
            model = YOLO(self.weights)                
            model.to(device)
            self.log(f"Model loaded on {device}")
            return model
        except Exception as e:
            self.handle_error(f"Error loading model: {e}")

    def get_frame(self):
        while True:
            try:
                self.ret, frame = self.cap.read()
                if self.ret:
                    self.lock.acquire()
                    self.current_frame = frame
                    self.lock.release()
            except Exception as e:
                self.handle_error(f"Error capturing frame: {e}")

    def start_detection(self):
        self.log("Starting detection")
        try:
            thread = threading.Thread(target=self.get_frame)
            thread.daemon = True
            thread.start()

            while True:
                if self.current_frame is not None:
                    self.lock.acquire()
                    frame = self.current_frame.copy()
                    self.lock.release()

                    try:
                        # Lakukan inferensi pada frame yang diambil
                        results = self.model.predict(source=frame, verbose=False)
                        
                        # Tampilkan frame dengan anotasi
                        annotated_frame = results[0].plot()
                        cv2.imshow('YOLOv8 Face Detection', annotated_frame)
                    except cv2.error as e:
                        self.handle_error(f"OpenCV error: {e}")

                # Tekan 'q' untuk keluar dari loop
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            self.cap.release()
            cv2.destroyAllWindows()
        except Exception as e:
            self.handle_error(f"Error during detection: {e}")

            self.log("Starting detection")
            try:
                thread = threading.Thread(target=self.get_frame)
                thread.daemon = True
                thread.start()

                while True:
                    if self.current_frame is not None:
                        self.lock.acquire()
                        frame = self.current_frame.copy()
                        self.lock.release()

                        # Lakukan inferensi pada frame yang diambil
                        results = self.model.predict(source=frame, verbose=False)

                        # Tampilkan frame dengan anotasi
                        annotated_frame = results[0].plot()
                        cv2.imshow('YOLOv8 Face Detection', annotated_frame)

                    # Tekan 'q' untuk keluar dari loop
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                self.cap.release()
                cv2.destroyAllWindows()
            except Exception as e:
                self.handle_error(f"Error during detection: {e}")
