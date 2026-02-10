"""
Hand Detection Module - Sử dụng MediaPipe để phát hiện bàn tay trong ROI của giáo viên
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
import time
from collections import deque, Counter

from shapely import buffer  # Thêm cho temporal smoothing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HandDetector:
    """
    Phát hiện và trích xuất thông tin bàn tay sử dụng MediaPipe
    Có temporal smoothing để giảm nhiễu
    """
    def __init__(self,
                static_mode: bool = False,
                max_hands: int = 2,
                min_detection_confidence: float = 0.7,
                min_tracking_confidence: float = 0.5,
                smoothing_window: int = 5):  # Thêm smoothing window
        """
        Khởi tạo hand detector
        
        Args:
            static_mode: Chế độ ảnh tĩnh hay video
            max_hands: Số lượng bàn tay tối đa cần phát hiện
            min_detection_confidence: Ngưỡng tin cậy cho detection
            min_tracking_confidence: Ngưỡng tin cậy cho tracking
            smoothing_window: Số frame dùng cho temporal smoothing (mặc định 5)
        """
        self.static_mode = static_mode
        self.max_hands = max_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.smoothing_window = smoothing_window
        
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_mode,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Temporal smoothing buffers - MỚI
        self.gesture_buffer = deque(maxlen=smoothing_window)  # Buffer cho gestures
        self.hand_count_buffer = deque(maxlen=smoothing_window)  # Buffer cho số tay
        self.bbox_buffer = {}  # {hand_index: deque of bboxes}
        
        logger.info("Hand Detector initialized with temporal smoothing")
        
    def find_hands(self, frame: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None,
                  draw: bool = True) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Phát hiện bàn tay trong frame hoặc ROI
        
        Args:
            frame: Frame gốc BGR
            roi: Region of Interest (x1, y1, x2, y2) nếu có
            draw: Có vẽ landmarks và kết nối hay không
            
        Returns:
            Tuple của (frame đã vẽ, danh sách thông tin bàn tay)
        """
        try:
            # Chuẩn bị frame và ROI
            if roi is not None:
                x1, y1, x2, y2 = roi
                # Mở rộng ROI để đảm bảo không bỏ sót bàn tay
                margin = int((x2 - x1) * 0.2)  # 20% margin
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(frame.shape[1], x2 + margin)
                y2 = min(frame.shape[0], y2 + margin)
                process_frame = frame[y1:y2, x1:x2]
            else:
                process_frame = frame
                x1, y1 = 0, 0
            
            # Chuyển sang RGB để xử lý
            frame_rgb = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
            
            # Phát hiện bàn tay
            results = self.hands.process(frame_rgb)
            
            # Frame để vẽ (copy để không ảnh hưởng input)
            if roi is not None:
                draw_frame = process_frame.copy()
            else:
                draw_frame = frame.copy()
                
            # Danh sách kết quả
            hands_info = []
            
            if results.multi_hand_landmarks:
                for hand_idx, (hand_landmarks, hand_handedness) in enumerate(
                    zip(results.multi_hand_landmarks, results.multi_handedness)):
                    
                    # Lấy thông tin bàn tay
                    hand_info = {
                        "landmarks": [],  # List of (x, y) normalized coordinates
                        "bbox": None,     # (x1, y1, x2, y2) in pixels
                        "label": hand_handedness.classification[0].label,  # "Left" or "Right"
                        "confidence": hand_handedness.classification[0].score
                    }
                    
                    # Tính bounding box
                    x_coords = []
                    y_coords = []
                    for landmark in hand_landmarks.landmark:
                        # Convert to pixel coordinates
                        px = int(landmark.x * process_frame.shape[1])
                        py = int(landmark.y * process_frame.shape[0])
                        x_coords.append(px)
                        y_coords.append(py)
                        # Save normalized coordinates
                        hand_info["landmarks"].append((landmark.x, landmark.y))
                    
                    # Calculate bbox with padding
                    padding = 20
                    bbox_x1 = max(0, min(x_coords) - padding)
                    bbox_y1 = max(0, min(y_coords) - padding)
                    bbox_x2 = min(process_frame.shape[1], max(x_coords) + padding)
                    bbox_y2 = min(process_frame.shape[0], max(y_coords) + padding)
                    
                    # Adjust bbox coordinates if using ROI
                    if roi is not None:
                        bbox_x1 += x1
                        bbox_y1 += y1
                        bbox_x2 += x1
                        bbox_y2 += y1
                    
                    hand_info["bbox"] = (bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                    
                    # Classify gesture - M\u1edaI
                    gesture = self.classify_simple_gesture(hand_landmarks)
                    hand_info["gesture"] = gesture
                    
                    if draw:
                        # Draw landmarks
                        self.mp_draw.draw_landmarks(
                            draw_frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_styles.get_default_hand_landmarks_style(),
                            self.mp_styles.get_default_hand_connections_style()
                        )
                        
                        # Draw bounding box
                        cv2.rectangle(draw_frame, (bbox_x1-x1, bbox_y1-y1), 
                                    (bbox_x2-x1, bbox_y2-y1), (0, 255, 0), 2)
                        
                        # Draw hand label
                        label = f"{hand_info['label']} ({hand_info['confidence']:.2f})"
                        # Thêm gesture vào label
                        if hand_info.get('gesture'):
                            label += f" - {hand_info['gesture']}"
                        cv2.putText(draw_frame, label, (bbox_x1-x1+5, bbox_y1-y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    hands_info.append(hand_info)
            
            # Copy kết quả vẽ vào frame gốc nếu dùng ROI
            if roi is not None:
                frame = frame.copy()
                frame[y1:y2, x1:x2] = draw_frame
                
            return frame, hands_info
            
        except Exception as e:
            logger.error(f"Error in hand detection: {e}")
            return frame, []
    
    def classify_simple_gesture(self, hand_landmarks) -> str:
        """
        Phân loại cử chỉ đơn giản dựa vào landmarks
        
        Returns:
            Tên cử chỉ: "open", "closed", "pointing", "peace", "thumbs_up", "unknown"
        """
        try:
            # Lấy landmarks
            landmarks = hand_landmarks.landmark
            
            # Định nghĩa các điểm quan trọng
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            ring_tip = landmarks[16]
            pinky_tip = landmarks[20]
            
            # Base của các ngón
            thumb_ip = landmarks[3]
            index_mcp = landmarks[5]
            middle_mcp = landmarks[9]
            ring_mcp = landmarks[13]
            pinky_mcp = landmarks[17]
            wrist = landmarks[0]
            
            # Kiểm tra ngón nào duỗi (tip cao hơn mcp/ip)
            def is_finger_extended(tip, mcp_or_ip, wrist_ref):
                # So sánh y coordinate (nhỏ hơn = cao hơn trong ảnh)
                return tip.y < mcp_or_ip.y
            
            thumb_extended = thumb_tip.x > thumb_ip.x if hand_landmarks else False  # Thumb đặc biệt
            index_extended = is_finger_extended(index_tip, index_mcp, wrist)
            middle_extended = is_finger_extended(middle_tip, middle_mcp, wrist)
            ring_extended = is_finger_extended(ring_tip, ring_mcp, wrist)
            pinky_extended = is_finger_extended(pinky_tip, pinky_mcp, wrist)
            
            # Đếm số ngón duỗi
            extended_count = sum([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended])
            
            # Phân loại cử chỉ
            if extended_count == 5:
                return "open"  # Bàn tay mở
            elif extended_count == 0:
                return "closed"  # Nắm đấm
            elif extended_count == 1 and index_extended:
                return "pointing"  # Chỉ tay
            elif extended_count == 2 and index_extended and middle_extended:
                return "peace"  # Cử chỉ peace/victory
            elif extended_count == 1 and thumb_extended:
                return "thumbs_up"  # Ngón cái lên
            else:
                return "unknown"
                
        except Exception as e:
            logger.error(f"Error classifying gesture: {e}")
            return "unknown"
    
    def smooth_gesture(self, current_gesture: str) -> str:
        """
        Làm mượt cử chỉ qua temporal buffer
        Sử dụng voting: chọn cử chỉ xuất hiện nhiều nhất trong buffer
        
        Args:
            current_gesture: Cử chỉ hiện tại
            
        Returns:
            Cử chỉ ổn định sau khi làm mượt
        """
        # Thêm vào buffer
        self.gesture_buffer.append(current_gesture)
        
        # Nếu buffer chưa đầy, trả về current
        if len(self.gesture_buffer) < self.smoothing_window // 2:
            return current_gesture
        
        # Voting: chọn gesture xuất hiện nhiều nhất
        counter = Counter(self.gesture_buffer)
        stable_gesture, count = counter.most_common(1)[0]
        
        # Chỉ công nhận nếu xuất hiện ít nhất 60% số frame
        if count >= len(self.gesture_buffer) * 0.6:
            return stable_gesture
        else:
            return "unknown"  # Không ổn định
    
    def smooth_hand_count(self, current_count: int) -> int:
        """
        Làm mượt số lượng tay phát hiện
        """
        self.hand_count_buffer.append(current_count)
        
        if len(self.hand_count_buffer) < self.smoothing_window // 2:
            return current_count
        
        # Lấy mode (giá trị xuất hiện nhiều nhất)
        counter = Counter(self.hand_count_buffer)
        stable_count, _ = counter.most_common(1)[0]
        return stable_count
    
    def get_stable_hands(self, frame: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None,
                        draw: bool = True) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Phiên bản có temporal smoothing của find_hands
        Trả về kết quả ổn định hơn, giảm nhiễu frame-by-frame
        
        Args:
            frame: Frame gốc BGR
            roi: Region of Interest (x1, y1, x2, y2) nếu có
            draw: Có vẽ landmarks và kết nối hay không
            
        Returns:
            Tuple của (frame đã vẽ, danh sách thông tin bàn tay ổn định)
        """
        # Detect hands thông thường
        frame, hands_info = self.find_hands(frame, roi, draw)
        
        # Smooth gesture cho mỗi tay
        for hand_info in hands_info:
            if 'gesture' in hand_info:
                original_gesture = hand_info['gesture']
                stable_gesture = self.smooth_gesture(original_gesture)
                hand_info['gesture'] = stable_gesture
                hand_info['gesture_confidence'] = 'stable' if stable_gesture != "unknown" else 'unstable'
        
        # Smooth hand count
        current_count = len(hands_info)
        stable_count = self.smooth_hand_count(current_count)
        
        # Chỉ trả về số tay ổn định
        # Nếu stable_count < current_count, loại bỏ tay có confidence thấp nhất
        if stable_count < current_count and hands_info:
            hands_info = sorted(hands_info, key=lambda h: h['confidence'], reverse=True)[:stable_count]
        
        return frame, hands_info
    
    def reset_buffers(self):
        """
        Reset các buffer (dùng khi thay đổi scene/context)
        """
        self.gesture_buffer.clear()
        self.hand_count_buffer.clear()
        self.bbox_buffer.clear()
        logger.info("Temporal buffers reset")
            
    def get_gesture_roi(self, frame: np.ndarray, hand_info: Dict[str, Any],
                       padding: int = 20) -> np.ndarray:
        """
        Cắt vùng ROI của bàn tay để nhận diện cử chỉ
        
        Args:
            frame: Frame gốc
            hand_info: Thông tin bàn tay từ find_hands
            padding: Padding thêm cho ROI
            
        Returns:
            ROI của bàn tay
        """
        if hand_info["bbox"] is None:
            return None
            
        x1, y1, x2, y2 = hand_info["bbox"]
        
        # Add padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame.shape[1], x2 + padding)
        y2 = min(frame.shape[0], y2 + padding)
        
        return frame[y1:y2, x1:x2]
        
    def cleanup(self):
        """Giải phóng tài nguyên"""
        self.hands.close()
        logger.info("Hand Detector cleaned up")