"""
ArcFace Recognition Module
"""

import torch
import cv2
import numpy as np
import pickle
import os
import logging
from typing import Optional, Tuple, List, Dict, Any


try:
    from insightface.app import FaceAnalysis
    from insightface.data import get_image as ins_get_image
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logging.warning("InsightFace not available. Run: pip install insightface onnxruntime-gpu")

logger = logging.getLogger(__name__)


class ArcFaceConfig:
    """Cấu hình cho ArcFace recognizer"""
    def __init__(self):
        self.model_name = 'buffalo_sc'  # Small model (16MB), nhanh nhất
        # Các lựa chọn khác:
        # - 'buffalo_l': Large model (nặng hơn, chính xác hơn)
        # - 'buffalo_s': Medium model (cân bằng)
        
        self.detection_threshold = 0.5  # Ngưỡng phát hiện khuôn mặt
        self.recognition_threshold = 0.5  # Ngưỡng nhận diện (cosine similarity)
        self.min_face_size = 20  # Kích thước khuôn mặt tối thiểu
        
        # Quality filtering
        self.use_quality_filter = True  # Lọc ảnh chất lượng thấp
        self.min_quality_score = 0.6  # Điểm chất lượng tối thiểu
        
        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 'cuda' hoặc 'cpu'
        self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']


class ArcFaceRecognizer:
    """
    Nhận diện khuôn mặt sử dụng InsightFace (ArcFace)
    Thay thế cho FaceNet - nhanh hơn và chính xác hơn
    """
    
    def __init__(self, config: ArcFaceConfig = None, database_path: str = "teacher_database_arcface.pkl"):
        if not INSIGHTFACE_AVAILABLE:
            raise ImportError("InsightFace không khả dụng. Cài: pip install insightface onnxruntime-gpu")
        
        self.config = config or ArcFaceConfig()
        self.database_path = database_path
        self.face_app = None
        self.teacher_database = {}  # {teacher_id: [embeddings]}
        self.is_initialized = False
        
        logger.info("ArcFace Recognizer initialized")
    
    def initialize(self) -> bool:
        """Khởi tạo InsightFace model"""
        try:
            logger.info(f"Loading InsightFace model: {self.config.model_name}")
            
            # Khởi tạo FaceAnalysis
            self.face_app = FaceAnalysis(
                name=self.config.model_name,
                providers=self.config.providers
            )
            
            # Prepare model với ctx_id (0=GPU, -1=CPU)
            ctx_id = 0 if 'cuda' in self.config.device.lower() else -1
            self.face_app.prepare(
                ctx_id=ctx_id,
                det_thresh=self.config.detection_threshold,
                det_size=(640, 640)  # Detection input size
            )
            
            # Load database nếu có
            self.load_database()
            
            self.is_initialized = True
            logger.info("InsightFace model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize InsightFace: {e}")
            return False
    
    def detect_and_extract(self, frame: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> List[Dict[str, Any]]:
        """
        Phát hiện khuôn mặt và trích xuất embedding trong frame hoặc ROI
        
        Args:
            frame: BGR frame từ OpenCV
            roi: Optional (x1, y1, x2, y2) để giới hạn vùng tìm kiếm
            
        Returns:
            List faces với thông tin:
            {
                'bbox': (x1, y1, x2, y2),
                'embedding': numpy array (512-dim),
                'det_score': float,  # Detection confidence
                'quality_score': float,  # Chất lượng ảnh (nếu có)
                'landmarks': numpy array (5 points: 2 eyes, nose, 2 mouth corners)
            }
        """
        if not self.is_initialized:
            logger.error("Model chưa được khởi tạo")
            return []
        
        try:
            # Nếu có ROI, crop frame
            if roi is not None:
                x1, y1, x2, y2 = roi
                x1, y1 = max(0, x1), max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                crop_frame = frame[y1:y2, x1:x2]
                offset_x, offset_y = x1, y1
            else:
                crop_frame = frame
                offset_x, offset_y = 0, 0
            
            # Chuyển sang RGB (InsightFace dùng RGB)
            rgb_frame = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB)
            
            # Detect và extract
            faces = self.face_app.get(rgb_frame)
            
            # Chuyển đổi sang format dễ dùng
            results = []
            for face in faces:
                # Bbox (adjust offset nếu có ROI)
                bbox = face.bbox.astype(int)
                bbox[0] += offset_x
                bbox[1] += offset_y
                bbox[2] += offset_x
                bbox[3] += offset_y
                
                # Embedding (đã normalize)
                embedding = face.normed_embedding  # 512-dim, normalized
                
                # Quality score (nếu model hỗ trợ)
                quality_score = getattr(face, 'embedding_norm', 1.0)
                
                # Detection confidence
                det_score = face.det_score
                
                # Landmarks (5 points)
                landmarks = face.landmark_2d_106 if hasattr(face, 'landmark_2d_106') else face.kps
                
                # Quality filtering
                if self.config.use_quality_filter:
                    if quality_score < self.config.min_quality_score:
                        logger.debug(f"Face rejected: quality {quality_score:.2f} < {self.config.min_quality_score}")
                        continue
                
                # Face size filtering
                face_width = bbox[2] - bbox[0]
                face_height = bbox[3] - bbox[1]
                if face_width < self.config.min_face_size or face_height < self.config.min_face_size:
                    logger.debug(f"Face rejected: size {face_width}x{face_height} < {self.config.min_face_size}")
                    continue
                
                results.append({
                    'bbox': tuple(bbox),
                    'embedding': embedding,
                    'det_score': float(det_score),
                    'quality_score': float(quality_score),
                    'landmarks': landmarks
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in face detection/extraction: {e}")
            return []
    
    def recognize_face(self, embedding: np.ndarray, threshold: Optional[float] = None) -> Tuple[Optional[str], float]:
        """
        Nhận diện khuôn mặt từ embedding
        
        Args:
            embedding: Face embedding (512-dim, normalized)
            threshold: Custom threshold (None = dùng config)
            
        Returns:
            (teacher_id, similarity_score) hoặc (None, best_score)
        """
        if not self.teacher_database:
            return None, 0.0
        
        threshold = threshold or self.config.recognition_threshold
        
        try:
            best_match = None
            best_similarity = -1.0
            
            for teacher_id, stored_embeddings in self.teacher_database.items():
                for stored_emb in stored_embeddings:
                    # Cosine similarity (cả 2 đã normalized)
                    similarity = float(np.dot(embedding, stored_emb))
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = teacher_id
            
            # Kiểm tra threshold
            if best_similarity >= threshold:
                return best_match, best_similarity
            else:
                return None, best_similarity
                
        except Exception as e:
            logger.error(f"Error in face recognition: {e}")
            return None, 0.0
    
    def load_database(self):
        """Load teacher database"""
        try:
            if os.path.exists(self.database_path):
                with open(self.database_path, 'rb') as f:
                    self.teacher_database = pickle.load(f)
                logger.info(f"Database loaded: {len(self.teacher_database)} teachers")
            else:
                logger.info("No database found, starting empty")
                self.teacher_database = {}
        except Exception as e:
            logger.error(f"Error loading database: {e}")
            self.teacher_database = {}
    
    def save_database(self):
        """Save teacher database"""
        try:
            with open(self.database_path, 'wb') as f:
                pickle.dump(self.teacher_database, f)
            logger.info(f"Database saved to {self.database_path}")
        except Exception as e:
            logger.error(f"Error saving database: {e}")
    
    def add_teacher(self, teacher_id: str, face_images: List[np.ndarray]) -> int:
        """
        Thêm giáo viên vào database
        
        Args:
            teacher_id: ID giáo viên
            face_images: List ảnh khuôn mặt (BGR frames)
            
        Returns:
            Số embeddings được thêm thành công
        """
        embeddings = []
        
        for img in face_images:
            faces = self.detect_and_extract(img)
            if faces:
                # Lấy face lớn nhất
                largest_face = max(faces, key=lambda f: (f['bbox'][2]-f['bbox'][0])*(f['bbox'][3]-f['bbox'][1]))
                embeddings.append(largest_face['embedding'])
        
        if embeddings:
            if teacher_id in self.teacher_database:
                self.teacher_database[teacher_id].extend(embeddings)
            else:
                self.teacher_database[teacher_id] = embeddings
            
            logger.info(f"Added {len(embeddings)} embeddings for {teacher_id}")
            return len(embeddings)
        
        return 0
    
    def get_database_info(self) -> Dict[str, Any]:
        """Lấy thông tin database"""
        return {
            'total_teachers': len(self.teacher_database),
            'teachers': list(self.teacher_database.keys()),
            'embeddings_per_teacher': {
                tid: len(embs) for tid, embs in self.teacher_database.items()
            }
        }
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'face_app') and self.face_app is not None:
            del self.face_app
        logger.info("ArcFace Recognizer cleaned up")


# Utility function
def create_arcface_recognizer(recognition_threshold: float = 0.5,
                              min_quality: float = 0.6) -> ArcFaceRecognizer:
    """
    Tạo ArcFace recognizer với config đơn giản
    """
    config = ArcFaceConfig()
    config.recognition_threshold = recognition_threshold
    config.min_quality_score = min_quality
    
    recognizer = ArcFaceRecognizer(config)
    return recognizer
