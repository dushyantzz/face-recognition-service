"""Preprocessing utilities for face images."""
import cv2
import numpy as np
from typing import Tuple, Optional


def align_face_5point(image: np.ndarray, landmarks: dict, output_size: Tuple[int, int] = (112, 112)) -> np.ndarray:
    """
    Align face using 5-point landmarks.
    
    Args:
        image: Input image
        landmarks: Dictionary with 5 landmarks (left_eye, right_eye, nose, left_mouth, right_mouth)
        output_size: Output image size
        
    Returns:
        Aligned face image
    """
    # Reference landmarks for output_size
    reference = np.array([
        [0.31556875, 0.4615741],   # left eye
        [0.68262291, 0.45983393],  # right eye
        [0.50026249, 0.64050536],  # nose
        [0.34947028, 0.82469195],  # left mouth
        [0.65343915, 0.82325089]   # right mouth
    ], dtype=np.float32)
    
    reference[:, 0] *= output_size[0]
    reference[:, 1] *= output_size[1]
    
    # Source landmarks
    source = np.array([
        landmarks['left_eye'],
        landmarks['right_eye'],
        landmarks['nose'],
        landmarks['left_mouth'],
        landmarks['right_mouth']
    ], dtype=np.float32)
    
    # Compute similarity transform
    tform = cv2.estimateAffinePartial2D(source, reference)[0]
    
    # Apply transformation
    aligned = cv2.warpAffine(image, tform, output_size, flags=cv2.INTER_LINEAR)
    
    return aligned


def crop_face(image: np.ndarray, bbox: list, margin: float = 0.2) -> np.ndarray:
    """
    Crop face from image with margin.
    
    Args:
        image: Input image
        bbox: Bounding box [x1, y1, x2, y2]
        margin: Margin ratio to add around face
        
    Returns:
        Cropped face image
    """
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    
    # Calculate margin
    face_w = x2 - x1
    face_h = y2 - y1
    margin_w = int(face_w * margin)
    margin_h = int(face_h * margin)
    
    # Apply margin
    x1 = max(0, x1 - margin_w)
    y1 = max(0, y1 - margin_h)
    x2 = min(w, x2 + margin_w)
    y2 = min(h, y2 + margin_h)
    
    return image[y1:y2, x1:x2]


def normalize_image(image: np.ndarray, mean: Tuple[float, float, float] = (127.5, 127.5, 127.5),
                   std: Tuple[float, float, float] = (128.0, 128.0, 128.0)) -> np.ndarray:
    """
    Normalize image.
    
    Args:
        image: Input image
        mean: Mean values for normalization
        std: Std values for normalization
        
    Returns:
        Normalized image
    """
    image = image.astype(np.float32)
    image = (image - mean) / std
    return image


def enhance_image(image: np.ndarray) -> np.ndarray:
    """
    Enhance image quality (contrast, brightness).
    
    Args:
        image: Input image
        
    Returns:
        Enhanced image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge channels
    lab = cv2.merge([l, a, b])
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return enhanced


def compute_face_quality(face_image: np.ndarray, bbox: list) -> float:
    """
    Compute face quality score.
    
    Args:
        face_image: Face image
        bbox: Bounding box
        
    Returns:
        Quality score (0-1)
    """
    # Size score (prefer larger faces)
    x1, y1, x2, y2 = bbox
    size = (x2 - x1) * (y2 - y1)
    size_score = min(size / (200 * 200), 1.0)  # Normalize by 200x200
    
    # Blur score (prefer sharper images)
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_score = min(blur / 500, 1.0)  # Normalize
    
    # Brightness score (prefer well-lit faces)
    brightness = np.mean(gray)
    brightness_score = 1.0 - abs(brightness - 128) / 128
    
    # Combined score
    quality = 0.4 * size_score + 0.4 * blur_score + 0.2 * brightness_score
    
    return quality


def filter_low_quality_faces(detections: list, quality_threshold: float = 0.5) -> list:
    """
    Filter out low quality face detections.
    
    Args:
        detections: List of face detections
        quality_threshold: Minimum quality score
        
    Returns:
        Filtered detections
    """
    filtered = []
    for det in detections:
        if 'quality' in det and det['quality'] >= quality_threshold:
            filtered.append(det)
        elif 'quality' not in det:
            # Keep if quality not computed
            filtered.append(det)
    
    return filtered
