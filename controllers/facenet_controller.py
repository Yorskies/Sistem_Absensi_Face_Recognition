import os
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from sklearn.preprocessing import LabelEncoder
import joblib
from PIL import Image
from torchvision import transforms


class FaceNetController:
    def __init__(self, dataset_dir="output_faces", embedding_dir="embeddings"):
        self.dataset_dir = dataset_dir  # Folder dataset
        self.embedding_dir = embedding_dir  # Folder untuk menyimpan embeddings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()  # Load model FaceNet
        self.model.to(self.device)

        # Pastikan direktori embeddings ada
        os.makedirs(self.embedding_dir, exist_ok=True)

    def load_model(self):
        """Memuat model FaceNet."""
        model_path = "models/20180402-114759-vggface2.pt"  # Path ke file model yang diunduh
        model = InceptionResnetV1(pretrained=None, classify=False).eval()  # Inisialisasi model tanpa pre-trained
        
        try:
            # Memuat state_dict dari file pre-trained
            state_dict = torch.load(model_path, map_location=self.device)
            
            # Filter key yang sesuai dengan struktur model
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
            
            # Memuat state_dict yang telah difilter
            model.load_state_dict(filtered_state_dict, strict=False)
            
            print("Model berhasil dimuat dengan state_dict yang sudah difilter.")
        except Exception as e:
            print(f"Terjadi kesalahan saat memuat model: {e}")
        
        return model


    def preprocess_image(self, img_path):
        """Preprocess image for FaceNet."""
        try:
            transform = transforms.Compose([
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            image = Image.open(img_path).convert("RGB")
            image = transform(image)
            return image.unsqueeze(0).to(self.device)  # Tambahkan batch dimension
        except Exception as e:
            print(f"Error preprocessing image {img_path}: {e}")
            return None

    def create_embeddings(self):
        """Create embeddings incrementally for new images."""
        embeddings = []
        labels = []

        # Load embeddings dan labels lama jika tersedia
        if os.path.exists(os.path.join(self.embedding_dir, "embeddings.npy")):
            old_embeddings = np.load(os.path.join(self.embedding_dir, "embeddings.npy"))
            old_labels = np.load(os.path.join(self.embedding_dir, "labels.npy"))
            embeddings.extend(old_embeddings)
            labels.extend(old_labels)

        # Load atau buat label encoder
        label_encoder_path = os.path.join(self.embedding_dir, "label_encoder.pkl")
        if os.path.exists(label_encoder_path):
            label_encoder = joblib.load(label_encoder_path)
        else:
            label_encoder = LabelEncoder()
            # Jika label encoder belum di-fit, inisialisasi dengan label lama
            if labels:
                label_encoder.fit(labels)

        # Proses dataset untuk menemukan gambar baru
        for person_folder in os.listdir(self.dataset_dir):
            person_path = os.path.join(self.dataset_dir, person_folder)
            if not os.path.isdir(person_path):
                continue  # Lewati jika bukan folder

            # Pastikan properti `classes_` ada di LabelEncoder
            if not hasattr(label_encoder, 'classes_'):
                label_encoder.classes_ = np.array([])

            # Tambahkan label baru jika belum ada
            if person_folder not in label_encoder.classes_:
                label_encoder.classes_ = np.append(label_encoder.classes_, person_folder)

            for img_file in os.listdir(person_path):
                img_path = os.path.join(person_path, img_file)
                embedding_file = os.path.join(self.embedding_dir, f"{img_file}.npy")

                # Periksa apakah embedding untuk gambar ini sudah ada
                if os.path.exists(embedding_file):
                    continue  # Lewati jika sudah ada

                # Preprocess dan buat embedding untuk gambar baru
                img_tensor = self.preprocess_image(img_path)
                if img_tensor is not None:
                    with torch.no_grad():
                        embedding = self.model(img_tensor).squeeze().cpu().numpy()
                        embeddings.append(embedding)
                        labels.append(person_folder)

                        # Simpan embedding individual untuk gambar ini
                        np.save(embedding_file, embedding)
                        print(f"Saved embedding for {img_file}")

        # Simpan embeddings, labels, dan label encoder
        self.save_embeddings(embeddings, labels, label_encoder)


    def save_embeddings(self, embeddings, labels, label_encoder):
        """Save embeddings, labels, and label encoder to disk."""
        try:
            embeddings = np.array(embeddings)
            labels = np.array(labels)

            # Simpan embeddings dan labels
            embeddings_file = os.path.join(self.embedding_dir, "embeddings.npy")
            labels_file = os.path.join(self.embedding_dir, "labels.npy")
            np.save(embeddings_file, embeddings)
            np.save(labels_file, labels)

            # Simpan label encoder
            label_encoder_path = os.path.join(self.embedding_dir, "label_encoder.pkl")
            joblib.dump(label_encoder, label_encoder_path)

            print("Embeddings and labels saved successfully.")
            print(f"Embeddings shape: {embeddings.shape}")
            print(f"Label mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
        except Exception as e:
            print(f"Error saving embeddings: {e}")

