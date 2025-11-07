"""RetinaFace detector for face detection with ONNX optimization."""
import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional
import onnxruntime as ort
from pathlib import Path


class RetinaFaceDetector:
    """Face detector using RetinaFace architecture optimized for CPU inference."""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 confidence_threshold: float = 0.8,
                 nms_threshold: float = 0.4,
                 use_onnx: bool = True):
        """
        Initialize RetinaFace detector.
        
        Args:
            model_path: Path to model weights (ONNX or PyTorch)
            confidence_threshold: Detection confidence threshold
            nms_threshold: Non-maximum suppression threshold
            use_onnx: Use ONNX runtime for inference
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.use_onnx = use_onnx
        self.device = 'cpu'
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            self._load_pretrained()
    
    def _load_pretrained(self):
        """Load pretrained RetinaFace model."""
        try:
            from retinaface import RetinaFace as RF
            self.model = RF
            self.use_library = True
            print("✓ Loaded RetinaFace pretrained model")
        except ImportError:
            # Fallback to custom implementation
            print("⚠ RetinaFace library not found, using fallback detector")
            self._init_fallback_detector()
    
    def _init_fallback_detector(self):
        """Initialize OpenCV DNN-based face detector as fallback."""
        # Load OpenCV's pretrained face detector
        model_file = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
        config_file = "deploy.prototxt"
        
        try:
            self.model = cv2.dnn.readNetFromCaffe(config_file, model_file)
            self.use_library = False
            print("✓ Loaded OpenCV DNN face detector")
        except:
            print("⚠ Using Haar Cascade as final fallback")
            self.model = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.use_library = False
    
    def load_model(self, model_path: str):
        """Load model from file."""
        if self.use_onnx and model_path.endswith('.onnx'):
            self.session = ort.InferenceSession(
                model_path,
                providers=['CPUExecutionProvider']
            )
            print(f"✓ Loaded ONNX model from {model_path}")
        else:
            self.model = torch.load(model_path, map_location=self.device)
            print(f"✓ Loaded PyTorch model from {model_path}")
    
    def detect_faces(self, image: np.ndarray) -> List[dict]:
        """
        Detect faces in image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of detected faces with bounding boxes and landmarks
        """
        if hasattr(self, 'use_library') and self.use_library:
            return self._detect_with_library(image)
        else:
            return self._detect_with_opencv(image)
    
    def _detect_with_library(self, image: np.ndarray) -> List[dict]:
        """Detect faces using RetinaFace library."""
        try:
            faces = self.model.detect_faces(image)
            
            results = []
            for key, face_data in faces.items():
                if face_data['score'] >= self.confidence_threshold:
                    bbox = face_data['facial_area']
                    landmarks = face_data['landmarks']
                    
                    results.append({
                        'bbox': [bbox[0], bbox[1], bbox[2], bbox[3]],
                        'confidence': face_data['score'],
                        'landmarks': landmarks
                    })
            
            return results
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def _detect_with_opencv(self, image: np.ndarray) -> List[dict]:
        """Detect faces using OpenCV DNN or Haar Cascade."""
        h, w = image.shape[:2]
        
        if isinstance(self.model, cv2.dnn.Net):
            # DNN detector
            blob = cv2.dnn.blobFromImage(
                image, 1.0, (300, 300), (104.0, 177.0, 123.0)
            )
            self.model.setInput(blob)
            detections = self.model.forward()
            
            results = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence >= self.confidence_threshold:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype(int)
                    
                    results.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(confidence),
                        'landmarks': self._estimate_landmarks(image, [x1, y1, x2, y2])
                    })
        else:
            # Haar Cascade
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.model.detectMultiScale(gray, 1.3, 5)
            
            results = []
            for (x, y, w, h) in faces:
                results.append({
                    'bbox': [x, y, x+w, y+h],
                    'confidence': 0.9,
                    'landmarks': self._estimate_landmarks(image, [x, y, x+w, y+h])
                })
        
        return results
    
    def _estimate_landmarks(self, image: np.ndarray, bbox: List[int]) -> dict:
        """Estimate 5-point facial landmarks from bounding box."""
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        
        # Approximate landmark positions
        landmarks = {
            'left_eye': (int(x1 + w * 0.3), int(y1 + h * 0.35)),
            'right_eye': (int(x1 + w * 0.7), int(y1 + h * 0.35)),
            'nose': (int(x1 + w * 0.5), int(y1 + h * 0.55)),
            'left_mouth': (int(x1 + w * 0.35), int(y1 + h * 0.75)),
            'right_mouth': (int(x1 + w * 0.65), int(y1 + h * 0.75))
        }
        
        return landmarks
    
    def extract_face_roi(self, image: np.ndarray, bbox: List[int], 
                        margin: float = 0.2) -> np.ndarray:
        """Extract face region of interest with margin."""
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        
        # Add margin
        margin_w = int((x2 - x1) * margin)
        margin_h = int((y2 - y1) * margin)
        
        x1 = max(0, x1 - margin_w)
        y1 = max(0, y1 - margin_h)
        x2 = min(w, x2 + margin_w)
        y2 = min(h, y2 + margin_h)
        
        return image[y1:y2, x1:x2]
    
    def benchmark(self, image: np.ndarray, num_runs: int = 100) -> dict:
        """Benchmark detection performance."""
        import time
        
        times = []
        for _ in range(num_runs):
            start = time.time()
            self.detect_faces(image)
            times.append(time.time() - start)
        
        return {
            'mean_latency_ms': np.mean(times) * 1000,
            'std_latency_ms': np.std(times) * 1000,
            'fps': 1.0 / np.mean(times)
        }
