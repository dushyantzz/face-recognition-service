"""Face detection endpoints"""

import base64
import io
import logging
from typing import List

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException
from PIL import Image

from src.api.schemas import DetectionRequest, DetectionResponse, Face
from src.api.app import detector

logger = logging.getLogger(__name__)
router = APIRouter()


def decode_image(file_content: bytes) -> np.ndarray:
    """Decode image from bytes"""
    try:
        img = Image.open(io.BytesIO(file_content))
        img_array = np.array(img)
        
        # Convert RGB to BGR for OpenCV
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        return img_array
    except Exception as e:
        logger.error(f"Failed to decode image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image format")


@router.post("/detect", response_model=DetectionResponse)
async def detect_faces(file: UploadFile = File(...)):
    """
    Detect faces in uploaded image
    
    Returns bounding boxes, landmarks, and confidence scores for all detected faces.
    """
    try:
        # Read and decode image
        content = await file.read()
        image = decode_image(content)
        
        logger.info(f"Processing image: {file.filename}, shape: {image.shape}")
        
        # Detect faces
        detections = detector.detect(image)
        
        if not detections:
            return DetectionResponse(
                face_count=0,
                faces=[],
                message="No faces detected"
            )
        
        # Convert detections to response format
        faces = []
        for det in detections:
            face = Face(
                bbox={
                    "x1": int(det['bbox'][0]),
                    "y1": int(det['bbox'][1]),
                    "x2": int(det['bbox'][2]),
                    "y2": int(det['bbox'][3])
                },
                confidence=float(det['confidence']),
                landmarks=det.get('landmarks')
            )
            faces.append(face)
        
        logger.info(f"Detected {len(faces)} face(s)")
        
        return DetectionResponse(
            face_count=len(faces),
            faces=faces,
            message="Success"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
