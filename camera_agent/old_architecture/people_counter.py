"""
People Counter Module - Đếm số lượng người trong video sử dụng YOLOv11n
Core module chỉ xử lý frame và đếm người, không bao gồm hiển thị
"""

import cv2
import numpy as np
import time
import logging
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
from ultralytics import YOLO
import threading

# Import ByteTrack wrapper
try:
    from byte_tracker_wrapper import SimpleByteTracker, TrackerConfig
    TRACKING_AVAILABLE = True
except ImportError:
    TRACKING_AVAILABLE = False
    logging.warning("ByteTrack wrapper not available. Tracking disabled.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PeopleCounterConfig:
    """Configuration for people counter"""
    model_name: str = "yolo11n.pt"  # YOLOv11 nano model
    confidence_threshold: float = 0.5  # Confidence threshold cho detection
    update_interval: float = 10.0  # Cập nhật mỗi 10 giây
    save_logs: bool = True  # Lưu log số lượng người
    # Preprocessing options
    enhance_image: bool = False  # Áp dụng image enhancement
    resize_factor: float = 1.0  # Scale factor cho input frame
    blur_reduction: bool = False  # Giảm blur
    contrast_enhancement: bool = False  # Tăng contrast
    # Tracking options - MỚI
    use_tracking: bool = True  # Bật ByteTrack tracking
    track_thresh: float = 0.5  # Ngưỡng confidence để track
    track_buffer: int = 30  # Số frame giữ track khi mất

class PeopleCounter:
    """
    Core module đếm số lượng người trong video
    Chỉ xử lý frame và detection, không bao gồm hiển thị
    """
    
    def __init__(self, config: PeopleCounterConfig = None):
        self.config = config or PeopleCounterConfig()
        self.model = None
        self.is_initialized = False
        self.is_active = False
        
        # Tracking variables
        self.current_count = 0
        self.last_count_time = time.time()
        self.count_history = []
        
        # ByteTrack tracker
        self.tracker = None
        if self.config.use_tracking and TRACKING_AVAILABLE:
            tracker_config = TrackerConfig()
            tracker_config.track_thresh = self.config.track_thresh
            tracker_config.track_buffer = self.config.track_buffer
            self.tracker = SimpleByteTracker(tracker_config)
            logger.info("ByteTrack tracking enabled")
        
        # Threading
        self.count_lock = threading.Lock()
        
        logger.info("People Counter initialized")
    
    def initialize(self) -> bool:
        """Khởi tạo YOLO model"""
        try:
            logger.info(f"Loading YOLO model: {self.config.model_name}")
            self.model = YOLO(self.config.model_name)
            self.is_initialized = True
            logger.info("YOLO model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize YOLO model: {e}")
            self.is_initialized = False
            return False
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Tiền xử lý frame để cải thiện chất lượng nhận diện
        
        Args:
            frame: Input frame từ video
            
        Returns:
            Preprocessed frame
        """
        if not self.config.enhance_image:
            return frame
        
        processed_frame = frame.copy()
        
        try:
            # Resize nếu cần
            if self.config.resize_factor != 1.0:
                new_width = int(frame.shape[1] * self.config.resize_factor)
                new_height = int(frame.shape[0] * self.config.resize_factor)
                processed_frame = cv2.resize(processed_frame, (new_width, new_height))
            
            # Giảm blur (sharpening)
            if self.config.blur_reduction:
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                processed_frame = cv2.filter2D(processed_frame, -1, kernel)
            
            # Tăng contrast và brightness
            if self.config.contrast_enhancement:
                # CLAHE (Contrast Limited Adaptive Histogram Equalization)
                lab = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2LAB)
                l_channel, a, b = cv2.split(lab)
                
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                l_channel = clahe.apply(l_channel)
                
                processed_frame = cv2.merge((l_channel, a, b))
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_LAB2BGR)
            
            return processed_frame
            
        except Exception as e:
            logger.warning(f"Error in frame preprocessing: {e}")
            return frame
    
    def detect_people(self, frame: np.ndarray) -> Tuple[int, List[Tuple[int, int, int, int, float]], Optional[List[Dict[str, Any]]]]:
        """
        Phát hiện và tracking người trong frame
        
        Args:
            frame: Input frame từ video
            
        Returns:
            Tuple containing (count, list of bounding boxes with confidence, tracked_objects)
            - tracked_objects là None nếu không dùng tracking
        """
        if not self.is_initialized or self.model is None:
            return 0, [], None
        
        try:
            # Tiền xử lý frame
            processed_frame = self.preprocess_frame(frame)
            
            # Chạy inference
            results = self.model(processed_frame, conf=self.config.confidence_threshold, verbose=False)
            
            people_boxes = []
            
            # Lọc chỉ class "person" (class 0 trong COCO dataset)
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Kiểm tra class là person (class 0)
                        if int(box.cls) == 0:  # person class
                            # Lấy tọa độ bounding box
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            confidence = float(box.conf[0])
                            
                            people_boxes.append((x1, y1, x2, y2, confidence))
            
            # Nếu có tracker, dùng tracking
            tracked_objects = None
            if self.tracker is not None:
                tracked_objects = self.tracker.update(people_boxes)
                people_count = len(tracked_objects)  # Đếm theo tracked objects
            else:
                people_count = len(people_boxes)  # Đếm trực tiếp
            
            return people_count, people_boxes, tracked_objects
            
        except Exception as e:
            logger.error(f"Error in people detection: {e}")
            return 0, [], None
    
    def should_update_count(self) -> bool:
        """Kiểm tra có nên cập nhật count không (dựa vào interval)"""
        current_time = time.time()
        return (current_time - self.last_count_time) >= self.config.update_interval
    
    def update_count(self, count: int):
        """Cập nhật số lượng người và lưu vào lịch sử"""
        with self.count_lock:
            current_time = time.time()
            
            # Chỉ cập nhật nếu đủ thời gian interval
            if self.should_update_count():
                self.current_count = count
                self.last_count_time = current_time
                
                # Lưu vào lịch sử
                if self.config.save_logs:
                    self.count_history.append({
                        'timestamp': current_time,
                        'count': count,
                        'time_str': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))
                    })
                
                logger.info(f"Updated people count: {count}")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[int, List[Tuple[int, int, int, int, float]], Optional[List[Dict[str, Any]]]]:
        """
        Xử lý frame: detect và track người
        
        Args:
            frame: Input frame từ video
            
        Returns:
            Tuple containing (people count, list of bounding boxes, tracked_objects)
        """
        if not self.is_initialized:
            return 0, [], None
        
        # Detect và track người
        count, boxes, tracked_objects = self.detect_people(frame)
        
        # Cập nhật count nếu cần
        if self.should_update_count():
            self.update_count(count)
        
        return count, boxes, tracked_objects
    
    def get_current_count(self) -> int:
        """Lấy số lượng người hiện tại (đã được cập nhật)"""
        with self.count_lock:
            return self.current_count
    
    def get_count_history(self) -> List[Dict[str, Any]]:
        """Lấy lịch sử số lượng người"""
        with self.count_lock:
            return self.count_history.copy()
    
    def reset_count_history(self):
        """Reset lịch sử đếm"""
        with self.count_lock:
            self.count_history.clear()
            logger.info("Count history reset")
    
    def save_count_log(self, filename: str):
        """Lưu lịch sử đếm ra file"""
        try:
            import json
            with self.count_lock:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.count_history, f, indent=2, ensure_ascii=False)
            logger.info(f"Count history saved to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save count log: {e}")
    
    def start(self):
        """Bắt đầu people counter"""
        if not self.initialize():
            logger.error("Failed to initialize people counter")
            return False
        
        self.is_active = True
        logger.info("People Counter started")
        return True
    
    def stop(self):
        """Dừng people counter"""
        self.is_active = False
        logger.info("People Counter stopped")
    
    def get_tracking_summary(self) -> Dict[str, Any]:
        """
        Lấy tổng kết tracking (entry/exit events)
        """
        if self.tracker is None:
            return {'tracking_enabled': False}
        
        summary = self.tracker.get_entry_exit_summary()
        summary['tracking_enabled'] = True
        return summary
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop()
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'tracker') and self.tracker is not None:
            self.tracker.reset()
        logger.info("People Counter cleaned up")

# Utility functions
def create_people_counter(confidence_threshold: float = 0.5, 
                         update_interval: float = 10.0,
                         enhance_image: bool = True) -> PeopleCounter:
    """
    Tạo PeopleCounter với config đơn giản
    
    Args:
        confidence_threshold: Ngưỡng confidence cho detection
        update_interval: Thời gian giữa các lần cập nhật count (giây)
        enhance_image: Có áp dụng image enhancement không
        
    Returns:
        PeopleCounter instance
    """
    config = PeopleCounterConfig(
        confidence_threshold=confidence_threshold,
        update_interval=update_interval,
        enhance_image=enhance_image
    )
    return PeopleCounter(config)

def process_video_file(video_path: str, 
                      counter: PeopleCounter,
                      start_frame: int = 0,
                      end_frame: int = None) -> List[Dict[str, Any]]:
    """
    Xử lý video file và trả về kết quả đếm người
    
    Args:
        video_path: Đường dẫn tới video file
        counter: PeopleCounter instance đã được khởi tạo
        start_frame: Frame bắt đầu xử lý
        end_frame: Frame kết thúc (None = xử lý hết video)
        
    Returns:
        List các kết quả detection theo frame
    """
    if not counter.is_initialized:
        logger.error("Counter chưa được khởi tạo")
        return []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Không thể mở video file: {video_path}")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame is None:
        end_frame = total_frames
    
    results = []
    frame_count = 0
    
    # Skip to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    try:
        while frame_count + start_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            people_count, boxes = counter.process_frame(frame)
            
            # Lưu kết quả
            results.append({
                'frame_number': frame_count + start_frame,
                'people_count': people_count,
                'detections': boxes,
                'timestamp': time.time()
            })
            
            frame_count += 1
            
            # Log progress
            if frame_count % 100 == 0:
                progress = ((frame_count + start_frame) / total_frames) * 100
                logger.info(f"Processed frame {frame_count + start_frame}/{total_frames} ({progress:.1f}%)")
    
    except Exception as e:
        logger.error(f"Error processing video: {e}")
    
    finally:
        cap.release()
    
    logger.info(f"Video processing completed. Processed {frame_count} frames")
    return results