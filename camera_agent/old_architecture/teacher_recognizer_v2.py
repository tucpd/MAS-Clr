"""
Teacher Recognition Module V2 - Nhận diện và tracking giáo viên trong video kết hợp face và person detection
"""

import cv2
import numpy as np
import time
import logging
import pickle
import os
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
import threading
import torch
from filterpy.kalman import KalmanFilter

try:
    from facenet_pytorch import InceptionResnetV1
    import torch
    from ultralytics import YOLO
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    logging.warning("Required packages not available. Please install: pip install facenet-pytorch torch torchvision ultralytics filterpy")

# Import ArcFace recognizer
try:
    from arcface_recognizer import ArcFaceRecognizer, ArcFaceConfig
    ARCFACE_AVAILABLE = True
except ImportError:
    ARCFACE_AVAILABLE = False
    logging.warning("ArcFace not available. Using FaceNet.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TeacherRecognizerConfig:
    """Configuration for teacher recognition"""
    database_path: str = "teacher_database.pkl"
    confidence_threshold: float = 0.9  # Face recognition confidence (cho FaceNet)
    arcface_threshold: float = 0.5  # Face recognition threshold cho ArcFace
    device: str = "auto"  # "cpu", "cuda", or "auto"
    min_face_size: int = 20
    face_margin: int = 10  # Margin around detected face
    person_confidence: float = 0.5  # Person detection confidence
    face_confidence: float = 0.5  # Face detection confidence
    tracking_timeout: float = 3.0  # Seconds until track is considered lost
    recognition_interval: float = 1.0  # Seconds between recognition attempts
    iou_threshold: float = 0.3  # IoU threshold for track matching
    # Backend selection - MỚI
    use_arcface: bool = True  # Dùng ArcFace (nhanh hơn) thay vì FaceNet
    arcface_model: str = "buffalo_sc"  # buffalo_sc (nhỏ), buffalo_s (medium), buffalo_l (lớn)

class PersonTrack:
    """Track information for a person"""
    def __init__(self, box: Tuple[int, int, int, int], current_time: float):
        self.box = box
        self.teacher_id = None
        self.confidence = 0.0
        self.last_seen = current_time
        self.kalman = self._init_kalman(box)
        self.history = []  # Track history for visualization
        self.embedding = None
        self.face_detected = False
        self.consecutive_misses = 0
        self.appearance_features = None
        
    def _init_kalman(self, box: Tuple[int, int, int, int]) -> KalmanFilter:
        """Initialize Kalman filter for box tracking"""
        kf = KalmanFilter(dim_x=8, dim_z=4)  # State: [x, y, w, h, vx, vy, vw, vh]
        
        # Initial state
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        kf.x = np.array([x1, y1, w, h, 0, 0, 0, 0])
        
        # State transition matrix
        kf.F = np.eye(8)
        kf.F[0, 4] = 1.0  # x velocity
        kf.F[1, 5] = 1.0  # y velocity
        kf.F[2, 6] = 1.0  # width velocity
        kf.F[3, 7] = 1.0  # height velocity
        
        # Measurement matrix
        kf.H = np.zeros((4, 8))
        kf.H[0, 0] = 1.0  # x
        kf.H[1, 1] = 1.0  # y
        kf.H[2, 2] = 1.0  # width
        kf.H[3, 3] = 1.0  # height
        
        # Covariance matrices
        kf.P *= 1000.0  # Initial state uncertainty
        kf.R = np.eye(4) * 100  # Measurement uncertainty
        kf.Q = np.eye(8) * 0.1  # Process uncertainty
        
        return kf
        
    def predict(self) -> Tuple[int, int, int, int]:
        """Predict next position using Kalman filter"""
        self.kalman.predict()
        x1 = int(self.kalman.x[0])
        y1 = int(self.kalman.x[1])
        w = int(self.kalman.x[2])
        h = int(self.kalman.x[3])
        return (x1, y1, x1 + w, y1 + h)
        
    def update(self, box: Tuple[int, int, int, int]):
        """Update track with new detection"""
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        measurement = np.array([x1, y1, w, h])
        self.kalman.update(measurement)
        self.box = box
        self.history.append(box)
        if len(self.history) > 30:  # Keep last 30 positions
            self.history.pop(0)
            
    def get_appearance_features(self, frame: np.ndarray) -> np.ndarray:
        """Extract appearance features from person ROI"""
        x1, y1, x2, y2 = self.box
        person_roi = frame[y1:y2, x1:x2]
        
        # Calculate color histogram
        hist_features = []
        for channel in cv2.split(person_roi):
            hist = cv2.calcHist([channel], [0], None, [32], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            hist_features.extend(hist)
            
        return np.array(hist_features)

class TeacherRecognizer:
    """
    Module nhận diện giáo viên trong video kết hợp face recognition và person tracking
    """
    
    def __init__(self, config: TeacherRecognizerConfig = None):
        self.config = config or TeacherRecognizerConfig()
        
        # Xác định device
        if self.config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device
        
        # Model components
        self.face_detector = None  # YOLOv11 for face detection (dùng với FaceNet)
        self.person_detector = None  # YOLOv11 for person detection
        self.face_encoder = None  # FaceNet for face embedding (nếu không dùng ArcFace)
        self.arcface_recognizer = None  # ArcFace recognizer (nếu được chọn)
        self.is_initialized = False
        self.use_arcface = self.config.use_arcface and ARCFACE_AVAILABLE
        
        # Database
        self.teacher_database = {}  # {teacher_id: [embeddings]}
        
        # Tracking
        self.tracks = {}  # {track_id: PersonTrack}
        self.next_track_id = 0
        self.last_recognition_time = 0
        
        # Threading
        self.recognition_lock = threading.Lock()
        
        logger.info(f"Teacher Recognizer V2 initialized (device: {self.device})")
        
    def initialize(self) -> bool:
        """Khởi tạo models"""
        if not MODELS_AVAILABLE:
            logger.error("Required packages not available. Please install required packages.")
            return False
        
        try:
            logger.info("Initializing detection and recognition models...")
            
            # Load YOLOv11 for person detection
            self.person_detector = YOLO('yolo11n.pt')  # Use small model for speed
            self.person_detector.conf = self.config.person_confidence
            
            if self.device == "cuda":
                self.person_detector.to(self.device)
            
            # Chọn backend: ArcFace hoặc FaceNet
            if self.use_arcface:
                logger.info("Using ArcFace backend (faster, more accurate)")
                arcface_config = ArcFaceConfig()
                arcface_config.model_name = self.config.arcface_model
                arcface_config.recognition_threshold = self.config.arcface_threshold
                arcface_config.detection_threshold = self.config.face_confidence
                arcface_config.device = self.device
                
                self.arcface_recognizer = ArcFaceRecognizer(
                    config=arcface_config,
                    database_path=self.config.database_path.replace('.pkl', '_arcface.pkl')
                )
                
                if not self.arcface_recognizer.initialize():
                    logger.error("Failed to initialize ArcFace, falling back to FaceNet")
                    self.use_arcface = False
            
            # Fallback hoặc nếu không dùng ArcFace
            if not self.use_arcface:
                logger.info("Using FaceNet backend (original)")
                # Load YOLOv11 for face detection
                face_model_path = "face_data/face_model.pt"
                if not os.path.exists(face_model_path):
                    logger.error(f"Face model not found at {face_model_path}")
                    return False
                    
                self.face_detector = YOLO(face_model_path)
                self.face_detector.conf = self.config.face_confidence
                
                if self.device == "cuda":
                    self.face_detector.to(self.device)
                
                # Initialize FaceNet
                self.face_encoder = InceptionResnetV1(
                    pretrained='vggface2',
                    device=self.device
                ).eval()
            
            # Load teacher database
            self.load_database()
            
            self.is_initialized = True
            logger.info(f"Models initialized successfully (backend: {'ArcFace' if self.use_arcface else 'FaceNet'})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            return False
            
    def detect_people(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect people in frame using YOLOv8"""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.person_detector(frame_rgb)
            result = results[0]
            
            person_boxes = []
            if result.boxes is not None:
                for box in result.boxes:
                    if box.cls == 0:  # Person class
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        person_boxes.append((x1, y1, x2, y2))
                        
            return person_boxes
            
        except Exception as e:
            logger.error(f"Error in person detection: {e}")
            return []
            
    def detect_faces(self, frame: np.ndarray, person_box: Tuple[int, int, int, int]) -> Tuple[Optional[torch.Tensor], Optional[Tuple[int, int, int, int]], Optional[np.ndarray], Optional[float]]:
        """
        Detect face within person bounding box
        Hỗ trợ cả ArcFace và FaceNet backend
        
        Returns:
            (face_tensor, face_box, embedding, quality_score)
            - face_tensor: Cho FaceNet (nếu dùng)
            - embedding: Cho ArcFace (nếu dùng)
            - quality_score: Điểm chất lượng ảnh
        """
        try:
            x1, y1, x2, y2 = person_box
            person_roi = frame[y1:y2, x1:x2]
            
            # Nếu dùng ArcFace - đơn giản hơn!
            if self.use_arcface:
                faces = self.arcface_recognizer.detect_and_extract(frame, roi=person_box)
                
                if not faces:
                    return None, None, None, None
                
                # Chọn face lớn nhất
                largest_face = max(faces, key=lambda f: (f['bbox'][2]-f['bbox'][0])*(f['bbox'][3]-f['bbox'][1]))
                
                return None, largest_face['bbox'], largest_face['embedding'], largest_face['quality_score']
            
            # Nếu dùng FaceNet - logic cũ
            else:
                # Detect faces in ROI
                results = self.face_detector(person_roi)
                result = results[0]
                
                if result.boxes is not None and len(result.boxes) > 0:
                    # Get largest face
                    areas = []
                    face_boxes = []
                    for box in result.boxes:
                        fx1, fy1, fx2, fy2 = map(int, box.xyxy[0])
                        area = (fx2 - fx1) * (fy2 - fy1)
                        areas.append(area)
                        face_boxes.append((fx1, fy1, fx2, fy2))
                        
                    if not face_boxes:
                        return None, None, None, None
                        
                    # Select largest face
                    largest_idx = np.argmax(areas)
                    fx1, fy1, fx2, fy2 = face_boxes[largest_idx]
                    
                    # Convert to global coordinates
                    global_box = (x1 + fx1, y1 + fy1, x1 + fx2, y1 + fy2)
                    
                    # Extract and preprocess face for FaceNet
                    face_roi = frame[global_box[1]:global_box[3], global_box[0]:global_box[2]]
                    face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                    face_resized = cv2.resize(face_rgb, (160, 160))
                    
                    # Convert to tensor
                    face_tensor = torch.from_numpy(face_resized).float() / 255.0
                    face_tensor = (face_tensor * 2.0) - 1.0  # Scale to [-1, 1]
                    face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0)
                    
                    if self.device == "cuda":
                        face_tensor = face_tensor.cuda()
                        
                    return face_tensor, global_box, None, None
                    
                return None, None, None, None
            
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return None, None, None, None
            
    def get_face_embedding(self, face_tensor: torch.Tensor = None, arcface_embedding: np.ndarray = None) -> Optional[np.ndarray]:
        """
        Generate face embedding
        Hỗ trợ cả FaceNet và ArcFace
        
        Args:
            face_tensor: FaceNet input (nếu dùng FaceNet)
            arcface_embedding: ArcFace embedding (nếu dùng ArcFace)
        """
        try:
            # Nếu dùng ArcFace, embedding đã có sẵn
            if self.use_arcface and arcface_embedding is not None:
                return arcface_embedding
            
            # Nếu dùng FaceNet, generate embedding
            if face_tensor is None:
                return None
                
            with torch.no_grad():
                embedding = self.face_encoder(face_tensor)
                return embedding.cpu().numpy().flatten()
                
        except Exception as e:
            logger.error(f"Error generating face embedding: {e}")
            return None
            
    def recognize_teacher(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Match face embedding with teacher database
        Hỗ trợ cả ArcFace và FaceNet
        """
        if len(self.teacher_database) == 0:
            return None, 0.0
            
        try:
            # Nếu dùng ArcFace, dùng recognizer của nó
            if self.use_arcface:
                return self.arcface_recognizer.recognize_face(embedding)
            
            # Nếu dùng FaceNet, logic cũ
            best_match = None
            best_confidence = 0.0
            
            for teacher_id, stored_embeddings in self.teacher_database.items():
                for stored_embedding in stored_embeddings:
                    confidence = self._calculate_similarity(embedding, stored_embedding)
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = teacher_id
                        
            if best_confidence >= self.config.confidence_threshold:
                return best_match, best_confidence
                
            return None, best_confidence
            
        except Exception as e:
            logger.error(f"Error in teacher recognition: {e}")
            return None, 0.0
            
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        try:
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            cosine_similarity = dot_product / (norm1 * norm2)
            return (cosine_similarity + 1) / 2  # Scale to [0, 1]
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
            
    def _calculate_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union between boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 < xi1 or yi2 < yi1:
            return 0.0
            
        intersection = (xi2 - xi1) * (yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0
        
    def update_tracks(self, frame: np.ndarray, person_boxes: List[Tuple[int, int, int, int]], current_time: float):
        """Update person tracks with new detections"""
        # Predict new positions for existing tracks
        predicted_boxes = {
            track_id: track.predict() 
            for track_id, track in self.tracks.items()
            if current_time - track.last_seen < self.config.tracking_timeout
        }
        
        # Match detections to existing tracks
        matched_tracks = {}
        matched_detections = set()
        
        for det_idx, det_box in enumerate(person_boxes):
            best_match = None
            best_iou = 0
            
            for track_id, pred_box in predicted_boxes.items():
                iou = self._calculate_iou(det_box, pred_box)
                if iou > best_iou and iou > self.config.iou_threshold:
                    best_iou = iou
                    best_match = track_id
                    
            if best_match is not None:
                matched_tracks[best_match] = det_idx
                matched_detections.add(det_idx)
                
        # Update matched tracks
        for track_id, det_idx in matched_tracks.items():
            track = self.tracks[track_id]
            det_box = person_boxes[det_idx]
            
            # Update Kalman filter
            track.update(det_box)
            track.last_seen = current_time
            track.consecutive_misses = 0
            
            # Try face detection if needed
            if not track.face_detected or current_time - self.last_recognition_time >= self.config.recognition_interval:
                face_tensor, face_box, arcface_embedding, quality_score = self.detect_faces(frame, det_box)
                
                if face_tensor is not None or arcface_embedding is not None:
                    track.face_detected = True
                    embedding = self.get_face_embedding(face_tensor, arcface_embedding)
                    
                    if embedding is not None:
                        track.embedding = embedding
                        
                        # Try recognition
                        teacher_id, confidence = self.recognize_teacher(embedding)
                        if teacher_id is not None:
                            track.teacher_id = teacher_id
                            track.confidence = confidence
                            self.last_recognition_time = current_time
                            
            # Update appearance features
            track.appearance_features = track.get_appearance_features(frame)
            
        # Create new tracks for unmatched detections
        for i, det_box in enumerate(person_boxes):
            if i not in matched_detections:
                track = PersonTrack(det_box, current_time)
                # Try immediate face detection
                face_tensor, face_box, arcface_embedding, quality_score = self.detect_faces(frame, det_box)
                if face_tensor is not None or arcface_embedding is not None:
                    track.face_detected = True
                    embedding = self.get_face_embedding(face_tensor, arcface_embedding)
                    if embedding is not None:
                        track.embedding = embedding
                        teacher_id, confidence = self.recognize_teacher(embedding)
                        if teacher_id is not None:
                            track.teacher_id = teacher_id
                            track.confidence = confidence
                            
                track.appearance_features = track.get_appearance_features(frame)
                self.tracks[self.next_track_id] = track
                self.next_track_id += 1
                
        # Update unmatched tracks
        for track_id in self.tracks:
            if track_id not in matched_tracks:
                self.tracks[track_id].consecutive_misses += 1
                
        # Remove old tracks
        self._remove_old_tracks(current_time)
        
    def _remove_old_tracks(self, current_time: float):
        """Remove tracks that haven't been seen recently"""
        track_ids = list(self.tracks.keys())
        for track_id in track_ids:
            track = self.tracks[track_id]
            if current_time - track.last_seen > self.config.tracking_timeout or track.consecutive_misses > 10:
                del self.tracks[track_id]
                
    def process_frame(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """Process frame and return detections with visualization"""
        if not self.is_initialized:
            return [], frame
            
        current_time = time.time()
        annotated_frame = frame.copy()
        
        try:
            # Detect people
            person_boxes = self.detect_people(frame)
            
            # Update tracking
            self.update_tracks(frame, person_boxes, current_time)
            
            # Prepare detections for visualization
            detections = []
            for track_id, track in self.tracks.items():
                if current_time - track.last_seen < 1.0:  # Only include recent tracks
                    detection = {
                        "track_id": track_id,
                        "box": track.box,
                        "teacher_id": track.teacher_id,
                        "confidence": track.confidence
                    }
                    detections.append(detection)
                    
                    # Draw track visualization
                    x1, y1, x2, y2 = track.box
                    
                    # Màu: Xanh lá nếu là giáo viên, xám nếu là học sinh
                    if track.teacher_id:
                        color = (0, 255, 0)  # Xanh lá - giáo viên
                    else:
                        color = (180, 180, 180)  # Xám - học sinh/người lạ
                    
                    # Draw person box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # CHỈ vẽ track history cho GIÁO VIÊN
                    # Học sinh không vẽ trail (chỉ có box)
                    if track.teacher_id and len(track.history) > 1:
                        # Vẽ đường di chuyển của giáo viên (10 vị trí gần nhất)
                        points = []
                        for hist_box in track.history[-10:]:
                            cx = (hist_box[0] + hist_box[2]) // 2  # Center X
                            cy = (hist_box[1] + hist_box[3]) // 2  # Center Y
                            points.append([cx, cy])
                        
                        if len(points) > 1:
                            points_array = np.array(points, dtype=np.int32)
                            cv2.polylines(annotated_frame, [points_array], False, (0, 255, 0), 2)
                    
                    # Draw label
                    if track.teacher_id:
                        label = f"{track.teacher_id} ({track.confidence:.2f})"
                    else:
                        label = f"Student"  # Đổi từ "Person" thành "Student" cho rõ ràng
                        
                    cv2.putText(annotated_frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                               color, 2)
            
            return detections, annotated_frame
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return [], annotated_frame
            
    def load_database(self):
        """Load teacher database from file"""
        try:
            # Nếu dùng ArcFace, dùng database của nó
            if self.use_arcface and self.arcface_recognizer:
                # ArcFace tự load database trong initialize()
                db_info = self.arcface_recognizer.get_database_info()
                logger.info(f"ArcFace database: {db_info['total_teachers']} teachers")
                # Sync với self.teacher_database để tương thích
                self.teacher_database = self.arcface_recognizer.teacher_database
            else:
                # FaceNet - logic cũ
                if os.path.exists(self.config.database_path):
                    with open(self.config.database_path, 'rb') as f:
                        self.teacher_database = pickle.load(f)
                    logger.info(f"Teacher database loaded with {len(self.teacher_database)} teachers")
                else:
                    self.teacher_database = {}
                    logger.info("No existing database found, starting empty")
                
        except Exception as e:
            logger.error(f"Error loading database: {e}")
            self.teacher_database = {}
            
    def get_database_info(self) -> Dict[str, Any]:
        """Get database statistics"""
        return {
            "total_teachers": len(self.teacher_database),
            "teachers": list(self.teacher_database.keys()),
            "embeddings_per_teacher": {
                teacher_id: len(embeddings) 
                for teacher_id, embeddings in self.teacher_database.items()
            }
        }
        
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'face_detector'):
            del self.face_detector
        if hasattr(self, 'person_detector'):
            del self.person_detector
        if hasattr(self, 'face_encoder'):
            del self.face_encoder
        logger.info("Teacher Recognizer cleaned up")