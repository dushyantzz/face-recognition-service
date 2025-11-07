"""Identity management endpoints"""

import io
import logging
import uuid
from typing import List, Optional
from datetime import datetime

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from PIL import Image

from src.api.schemas import IdentityResponse, IdentityListResponse, Identity
from src.api.app import detector, embedder, matcher, db_manager
from src.utils.preprocessing import align_face
from src.config import settings

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


@router.post("/identities", response_model=IdentityResponse)
async def add_identity(
    name: str = Form(...),
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None)
):
    """
    Add a new identity to the gallery
    
    Uploads an image, detects the face, extracts embedding, and stores in database.
    """
    try:
        # Read and decode image
        content = await file.read()
        image = decode_image(content)
        
        logger.info(f"Adding identity: {name}")
        
        # Detect face
        detections = detector.detect(image)
        
        if not detections:
            raise HTTPException(status_code=400, detail="No face detected in image")
        
        if len(detections) > 1:
            logger.warning(f"Multiple faces detected, using the first one")
        
        det = detections[0]
        
        # Align face
        aligned_face = align_face(
            image,
            det['landmarks'],
            output_size=(112, 112)
        )
        
        # Extract embedding
        embedding = embedder.extract_embedding(aligned_face)
        
        # Generate unique ID
        identity_id = str(uuid.uuid4())
        
        # Save aligned face image
        image_path = settings.GALLERY_DIR / f"{identity_id}.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_path), aligned_face)
        
        # Store in database
        await db_manager.add_identity(
            identity_id=identity_id,
            name=name,
            embedding=embedding,
            image_path=str(image_path),
            metadata=metadata
        )
        
        # Add to matcher
        matcher.add_embedding(identity_id, embedding, name)
        
        logger.info(f"Successfully added identity: {name} (ID: {identity_id})")
        
        return IdentityResponse(
            identity_id=identity_id,
            name=name,
            image_path=str(image_path),
            metadata=metadata,
            created_at=datetime.utcnow().isoformat(),
            message="Identity added successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add identity: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to add identity: {str(e)}")


@router.get("/identities", response_model=IdentityListResponse)
async def list_identities():
    """
    List all identities in the gallery
    """
    try:
        identities_data = await db_manager.get_all_identities()
        
        identities = []
        for data in identities_data:
            identities.append(Identity(
                identity_id=data['id'],
                name=data['name'],
                image_path=data.get('image_path'),
                metadata=data.get('metadata'),
                created_at=data.get('created_at', '')
            ))
        
        return IdentityListResponse(
            total=len(identities),
            identities=identities
        )
        
    except Exception as e:
        logger.error(f"Failed to list identities: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list identities: {str(e)}")


@router.get("/identities/{identity_id}", response_model=Identity)
async def get_identity(identity_id: str):
    """
    Get identity by ID
    """
    try:
        data = await db_manager.get_identity(identity_id)
        
        if not data:
            raise HTTPException(status_code=404, detail="Identity not found")
        
        return Identity(
            identity_id=data['id'],
            name=data['name'],
            image_path=data.get('image_path'),
            metadata=data.get('metadata'),
            created_at=data.get('created_at', '')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get identity: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get identity: {str(e)}")


@router.delete("/identities/{identity_id}")
async def delete_identity(identity_id: str):
    """
    Delete identity by ID
    """
    try:
        # Remove from database
        await db_manager.delete_identity(identity_id)
        
        # Remove from matcher
        matcher.remove_embedding(identity_id)
        
        logger.info(f"Deleted identity: {identity_id}")
        
        return {"message": "Identity deleted successfully", "identity_id": identity_id}
        
    except Exception as e:
        logger.error(f"Failed to delete identity: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete identity: {str(e)}")
