"""ArcFace/AdaFace embedding extractor with ONNX optimization."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, List
import onnxruntime as ort
from pathlib import Path


class ArcFaceEmbedder:
    """Face embedding extractor using ArcFace/AdaFace architecture."""
    
    def __init__(self,
                 model_path: Optional[str] = None,
                 model_name: str = 'arcface',
                 use_onnx: bool = True,
                 embedding_size: int = 512):
        """
        Initialize face embedding extractor.
        
        Args:
            model_path: Path to model weights
            model_name: Model architecture ('arcface' or 'adaface')
            use_onnx: Use ONNX runtime for inference
            embedding_size: Size of embedding vector
        """
        self.model_name = model_name
        self.use_onnx = use_onnx
        self.embedding_size = embedding_size
        self.device = 'cpu'
        self.input_size = (112, 112)
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            self._load_pretrained()
    
    def _load_pretrained(self):
        """Load pretrained model."""
        try:
            # Try loading with InsightFace
            import insightface
            from insightface.app import FaceAnalysis
            
            self.app = FaceAnalysis(
                name='buffalo_l',
                providers=['CPUExecutionProvider']
            )
            self.app.prepare(ctx_id=-1, det_size=(640, 640))
            self.use_library = True
            print(f"✓ Loaded {self.model_name} pretrained model")
            
        except Exception as e:
            print(f"⚠ Could not load pretrained model: {e}")
            print("⚠ Using fallback embedding extractor")
            self._init_fallback()
    
    def _init_fallback(self):
        """Initialize fallback embedding extractor."""
        try:
            from facenet_pytorch import InceptionResnetV1
            self.model = InceptionResnetV1(pretrained='vggface2').eval()
            self.use_library = False
            print("✓ Loaded FaceNet fallback model")
        except:
            print("⚠ Using simple CNN as final fallback")
            self.model = self._create_simple_cnn()
            self.use_library = False
    
    def _create_simple_cnn(self):
        """Create simple CNN for embeddings as final fallback."""
        import torch.nn as nn
        
        class SimpleFaceNet(nn.Module):
            def __init__(self, embedding_size=512):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                self.fc = nn.Linear(128, embedding_size)
            
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return F.normalize(x, p=2, dim=1)
        
        model = SimpleFaceNet(self.embedding_size)
        model.eval()
        return model
    
    def load_model(self, model_path: str):
        """Load model from file."""
        if self.use_onnx and model_path.endswith('.onnx'):
            self.session = ort.InferenceSession(
                model_path,
                providers=['CPUExecutionProvider']
            )
            self.use_onnx_model = True
            print(f"✓ Loaded ONNX model from {model_path}")
        else:
            self.model = torch.load(model_path, map_location=self.device)
            self.model.eval()
            self.use_onnx_model = False
            print(f"✓ Loaded PyTorch model from {model_path}")
    
    def preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """Preprocess face image for embedding extraction."""
        # Resize to input size
        face_image = cv2.resize(face_image, self.input_size)
        
        # Convert BGR to RGB
        if len(face_image.shape) == 3 and face_image.shape[2] == 3:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Normalize
        face_image = face_image.astype(np.float32)
        face_image = (face_image - 127.5) / 128.0
        
        # Transpose to CHW format
        face_image = np.transpose(face_image, (2, 0, 1))
        
        # Add batch dimension
        face_image = np.expand_dims(face_image, axis=0)
        
        return face_image
    
    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract embedding from face image.
        
        Args:
            face_image: Face image (BGR format)
            
        Returns:
            Normalized embedding vector
        """
        if hasattr(self, 'use_library') and self.use_library:
            return self._extract_with_insightface(face_image)
        elif hasattr(self, 'use_onnx_model') and self.use_onnx_model:
            return self._extract_with_onnx(face_image)
        else:
            return self._extract_with_pytorch(face_image)
    
    def _extract_with_insightface(self, face_image: np.ndarray) -> np.ndarray:
        """Extract embedding using InsightFace."""
        try:
            faces = self.app.get(face_image)
            if len(faces) > 0:
                # Return embedding from first detected face
                embedding = faces[0].embedding
                # Normalize
                embedding = embedding / np.linalg.norm(embedding)
                return embedding
            else:
                # No face detected, return zero embedding
                return np.zeros(self.embedding_size, dtype=np.float32)
        except Exception as e:
            print(f"Embedding extraction error: {e}")
            return np.zeros(self.embedding_size, dtype=np.float32)
    
    def _extract_with_onnx(self, face_image: np.ndarray) -> np.ndarray:
        """Extract embedding using ONNX model."""
        input_data = self.preprocess(face_image)
        
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        
        embedding = self.session.run(
            [output_name],
            {input_name: input_data}
        )[0]
        
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.flatten()
    
    def _extract_with_pytorch(self, face_image: np.ndarray) -> np.ndarray:
        """Extract embedding using PyTorch model."""
        input_data = self.preprocess(face_image)
        input_tensor = torch.from_numpy(input_data)
        
        with torch.no_grad():
            embedding = self.model(input_tensor)
        
        embedding = embedding.cpu().numpy()
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.flatten()
    
    def extract_batch_embeddings(self, face_images: List[np.ndarray]) -> np.ndarray:
        """Extract embeddings for batch of face images."""
        embeddings = []
        for face_image in face_images:
            embedding = self.extract_embedding(face_image)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def align_face(self, image: np.ndarray, landmarks: dict) -> np.ndarray:
        """Align face using 5-point landmarks."""
        # Reference landmarks (normalized coordinates for 112x112 image)
        reference_landmarks = np.array([
            [38.2946, 51.6963],  # left eye
            [73.5318, 51.5014],  # right eye
            [56.0252, 71.7366],  # nose
            [41.5493, 92.3655],  # left mouth
            [70.7299, 92.2041]   # right mouth
        ], dtype=np.float32)
        
        # Get source landmarks
        if isinstance(landmarks, dict):
            source_landmarks = np.array([
                landmarks.get('left_eye', [0, 0]),
                landmarks.get('right_eye', [0, 0]),
                landmarks.get('nose', [0, 0]),
                landmarks.get('left_mouth', [0, 0]),
                landmarks.get('right_mouth', [0, 0])
            ], dtype=np.float32)
        else:
            source_landmarks = np.array(landmarks, dtype=np.float32)
        
        # Compute transformation matrix
        tform = cv2.estimateAffinePartial2D(
            source_landmarks,
            reference_landmarks
        )[0]
        
        # Apply transformation
        aligned_face = cv2.warpAffine(
            image,
            tform,
            self.input_size,
            flags=cv2.INTER_LINEAR
        )
        
        return aligned_face
    
    def benchmark(self, face_image: np.ndarray, num_runs: int = 100) -> dict:
        """Benchmark embedding extraction performance."""
        import time
        
        times = []
        for _ in range(num_runs):
            start = time.time()
            self.extract_embedding(face_image)
            times.append(time.time() - start)
        
        return {
            'mean_latency_ms': np.mean(times) * 1000,
            'std_latency_ms': np.std(times) * 1000,
            'embeddings_per_second': 1.0 / np.mean(times)
        }
