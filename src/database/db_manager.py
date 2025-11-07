"""Database manager for storing face embeddings and metadata."""
import json
import numpy as np
from datetime import datetime
from typing import List, Optional, Dict
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, LargeBinary, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pathlib import Path

Base = declarative_base()


class FaceIdentity(Base):
    """Face identity model."""
    __tablename__ = 'face_identities'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    identity_id = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255))
    embedding = Column(LargeBinary, nullable=False)  # Store as binary
    image_path = Column(String(512))
    metadata_json = Column(Text)  # Store additional metadata as JSON
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'identity_id': self.identity_id,
            'name': self.name,
            'image_path': self.image_path,
            'metadata': json.loads(self.metadata_json) if self.metadata_json else {},
            'confidence': self.confidence,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class FaceDetection(Base):
    """Face detection log model."""
    __tablename__ = 'face_detections'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    identity_id = Column(String(255), index=True)
    frame_id = Column(String(255))
    bbox_x1 = Column(Integer)
    bbox_y1 = Column(Integer)
    bbox_x2 = Column(Integer)
    bbox_y2 = Column(Integer)
    confidence = Column(Float)
    similarity_score = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    metadata_json = Column(Text)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'identity_id': self.identity_id,
            'frame_id': self.frame_id,
            'bbox': [self.bbox_x1, self.bbox_y1, self.bbox_x2, self.bbox_y2],
            'confidence': self.confidence,
            'similarity_score': self.similarity_score,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'metadata': json.loads(self.metadata_json) if self.metadata_json else {}
        }


class DatabaseManager:
    """Database manager for face recognition service."""
    
    def __init__(self, db_url: str = 'sqlite:///face_recognition.db'):
        """
        Initialize database manager.
        
        Args:
            db_url: Database URL (SQLite, PostgreSQL, etc.)
        """
        self.engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        print(f"✓ Initialized database: {db_url}")
    
    def get_session(self) -> Session:
        """Get database session."""
        return self.SessionLocal()
    
    def add_identity(self,
                    identity_id: str,
                    embedding: np.ndarray,
                    name: Optional[str] = None,
                    image_path: Optional[str] = None,
                    metadata: Optional[dict] = None,
                    confidence: Optional[float] = None) -> bool:
        """
        Add face identity to database.
        
        Args:
            identity_id: Unique identifier
            embedding: Face embedding vector
            name: Person name
            image_path: Path to image file
            metadata: Additional metadata
            confidence: Detection confidence
            
        Returns:
            Success status
        """
        session = self.get_session()
        try:
            # Convert embedding to binary
            embedding_binary = embedding.astype(np.float32).tobytes()
            
            # Check if identity already exists
            existing = session.query(FaceIdentity).filter_by(identity_id=identity_id).first()
            if existing:
                # Update existing
                existing.embedding = embedding_binary
                existing.name = name
                existing.image_path = image_path
                existing.metadata_json = json.dumps(metadata) if metadata else None
                existing.confidence = confidence
                existing.updated_at = datetime.utcnow()
                print(f"✓ Updated identity: {identity_id}")
            else:
                # Create new
                identity = FaceIdentity(
                    identity_id=identity_id,
                    name=name,
                    embedding=embedding_binary,
                    image_path=image_path,
                    metadata_json=json.dumps(metadata) if metadata else None,
                    confidence=confidence
                )
                session.add(identity)
                print(f"✓ Added identity: {identity_id}")
            
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            print(f"❌ Error adding identity: {e}")
            return False
        finally:
            session.close()
    
    def get_identity(self, identity_id: str) -> Optional[dict]:
        """Get identity by ID."""
        session = self.get_session()
        try:
            identity = session.query(FaceIdentity).filter_by(identity_id=identity_id).first()
            if identity:
                result = identity.to_dict()
                # Add embedding
                embedding = np.frombuffer(identity.embedding, dtype=np.float32)
                result['embedding'] = embedding
                return result
            return None
        finally:
            session.close()
    
    def get_all_identities(self) -> List[dict]:
        """Get all identities."""
        session = self.get_session()
        try:
            identities = session.query(FaceIdentity).all()
            return [identity.to_dict() for identity in identities]
        finally:
            session.close()
    
    def get_all_embeddings(self) -> tuple:
        """Get all embeddings and identity IDs."""
        session = self.get_session()
        try:
            identities = session.query(FaceIdentity).all()
            embeddings = []
            identity_ids = []
            metadata_list = []
            
            for identity in identities:
                embedding = np.frombuffer(identity.embedding, dtype=np.float32)
                embeddings.append(embedding)
                identity_ids.append(identity.identity_id)
                metadata_list.append({
                    'name': identity.name,
                    'image_path': identity.image_path,
                    'metadata': json.loads(identity.metadata_json) if identity.metadata_json else {}
                })
            
            return np.array(embeddings), identity_ids, metadata_list
        finally:
            session.close()
    
    def remove_identity(self, identity_id: str) -> bool:
        """Remove identity from database."""
        session = self.get_session()
        try:
            identity = session.query(FaceIdentity).filter_by(identity_id=identity_id).first()
            if identity:
                session.delete(identity)
                session.commit()
                print(f"✓ Removed identity: {identity_id}")
                return True
            return False
        except Exception as e:
            session.rollback()
            print(f"❌ Error removing identity: {e}")
            return False
        finally:
            session.close()
    
    def log_detection(self,
                     identity_id: Optional[str],
                     bbox: List[int],
                     confidence: float,
                     similarity_score: Optional[float] = None,
                     frame_id: Optional[str] = None,
                     metadata: Optional[dict] = None) -> bool:
        """Log face detection event."""
        session = self.get_session()
        try:
            detection = FaceDetection(
                identity_id=identity_id,
                frame_id=frame_id,
                bbox_x1=bbox[0],
                bbox_y1=bbox[1],
                bbox_x2=bbox[2],
                bbox_y2=bbox[3],
                confidence=confidence,
                similarity_score=similarity_score,
                metadata_json=json.dumps(metadata) if metadata else None
            )
            session.add(detection)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            print(f"❌ Error logging detection: {e}")
            return False
        finally:
            session.close()
    
    def get_detection_history(self,
                            identity_id: Optional[str] = None,
                            limit: int = 100) -> List[dict]:
        """Get detection history."""
        session = self.get_session()
        try:
            query = session.query(FaceDetection)
            if identity_id:
                query = query.filter_by(identity_id=identity_id)
            query = query.order_by(FaceDetection.timestamp.desc()).limit(limit)
            
            detections = query.all()
            return [detection.to_dict() for detection in detections]
        finally:
            session.close()
    
    def get_statistics(self) -> dict:
        """Get database statistics."""
        session = self.get_session()
        try:
            total_identities = session.query(FaceIdentity).count()
            total_detections = session.query(FaceDetection).count()
            
            return {
                'total_identities': total_identities,
                'total_detections': total_detections
            }
        finally:
            session.close()
