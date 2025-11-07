"""Face recognition endpoints"""

import base64
import io
import logging
from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from PIL import Image

from src.api.schemas import RecognitionResponse, RecognizedFace, Match
from src.api.app import detector, embedder, matcher, db_manager
from src.utils.preprocessing import align_face

logger = logging.getLogger(__name__)
router = APIRouter()


def decode_image(file_content: bytes) -> np.ndarray:
    """Decode image from bytes"""
    try:
        img = Image.open(io.BytesIO(file_content))
        img_array = np.array(img)
        
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        return img_array
    except Exception as e:
        logger.error(f"Failed to decode image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image format")


@router.post("/recognize", response_model=RecognitionResponse)
async def recognize_faces(
    file: UploadFile = File(...),
    top_k: int = Query(5, ge=1, le=10, description="Number of top matches to return")
):
    """
    Recognize faces in uploaded image
    
    Detects faces, extracts embeddings, and matches against known identities.
    """
    try:
        # Read and decode image
        content = await file.read()
        image = decode_image(content)
        
        logger.info(f"Processing image for recognition: {file.filename}")
        
        # Detect faces
        detections = detector.detect(image)
        
        if not detections:
            return RecognitionResponse(
                face_count=0,
                recognized_faces=[],
                message="No faces detected"
            )
        
        recognized_faces = []
        
        for i, det in enumerate(detections):
            try:
                # Align face
                aligned_face = align_face(
                    image,
                    det['landmarks'],
                    output_size=(112, 112)
                )
                
                # Extract embedding
                embedding = embedder.extract_embedding(aligned_face)
                
                # Match against gallery
                matches = matcher.match(embedding, top_k=top_k)
                
                # Convert matches to response format
                match_list = []
                for match in matches:
                    match_list.append(Match(
                        identity_id=match['id'],
                        name=match['name'],
                        similarity=float(match['similarity']),
                        distance=float(match['distance'])
                    ))
                
                recognized_face = RecognizedFace(
                    face_index=i,
                    bbox={
                        "x1": int(det['bbox'][0]),
                        "y1": int(det['bbox'][1]),
                        "x2": int(det['bbox'][2]),
                        "y2": int(det['bbox'][3])
                    },
                    confidence=float(det['confidence']),
                    matches=match_list,
                    best_match=match_list[0] if match_list else None
                )
                
                recognized_faces.append(recognized_face)
                
            except Exception as e:
                logger.error(f"Failed to process face {i}: {e}")
                continue
        
        logger.info(f"Recognized {len(recognized_faces)} face(s)")
        
        return RecognitionResponse(
            face_count=len(recognized_faces),
            recognized_faces=recognized_faces,
            message="Success"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recognition error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Recognition failed: {str(e)}")
