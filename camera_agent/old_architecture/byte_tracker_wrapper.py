"""
ByteTrack Wrapper - Tracking wrapper cho YOLO detections
Đơn giản hóa việc sử dụng ByteTrack với YOLO
"""

import numpy as np
from typing import List, Tuple, Dict, Any
import logging

try:
    from yolox.tracker.byte_tracker import BYTETracker, STrack
    BYTETRACK_AVAILABLE = True
except ImportError:
    BYTETRACK_AVAILABLE = False
    logging.warning("ByteTrack not available. Run: pip install yolox")

logger = logging.getLogger(__name__)


class TrackerConfig:
    """Cấu hình cho ByteTracker"""
    def __init__(self):
        self.track_thresh = 0.5  # Ngưỡng confidence để bắt đầu track
        self.track_buffer = 30  # Số frame giữ track sau khi mất
        self.match_thresh = 0.8  # Ngưỡng IoU để match
        self.frame_rate = 30  # FPS của video


class SimpleByteTracker:
    """
    Wrapper đơn giản cho ByteTrack
    Chuyển đổi YOLO detections thành tracked objects
    """
    
    def __init__(self, config: TrackerConfig = None):
        if not BYTETRACK_AVAILABLE:
            raise ImportError("ByteTrack không khả dụng. Cài: pip install yolox")
        
        self.config = config or TrackerConfig()
        self.tracker = BYTETracker(
            track_thresh=self.config.track_thresh,
            track_buffer=self.config.track_buffer,
            match_thresh=self.config.match_thresh,
            frame_rate=self.config.frame_rate
        )
        
        # Lưu trữ lịch sử
        self.track_history = {}  # {track_id: [boxes]}
        self.current_tracks = []
        self.lost_tracks = []
        
        # Stats
        self.total_tracks_created = 0
        self.entry_exit_events = []  # Danh sách sự kiện vào/ra
        
        logger.info("ByteTracker initialized")
    
    def update(self, detections: List[Tuple[int, int, int, int, float]]) -> List[Dict[str, Any]]:
        """
        Cập nhật tracker với detections mới
        
        Args:
            detections: List của (x1, y1, x2, y2, confidence)
            
        Returns:
            List tracked objects với thông tin:
            {
                'track_id': int,
                'bbox': (x1, y1, x2, y2),
                'confidence': float,
                'tlwh': (x, y, w, h),  # Top-left-width-height format
                'is_activated': bool,
                'tracklet_len': int  # Số frame đã track
            }
        """
        if not detections:
            # Không có detection, chỉ propagate tracks cũ
            online_targets = self.tracker.update(
                np.empty((0, 5)),  # Empty array
                [0, 0],  # Dummy image size
                [0, 0]   # Dummy test size
            )
        else:
            # Chuyển sang định dạng ByteTrack: [x1, y1, x2, y2, conf]
            det_array = np.array(detections)
            
            # ByteTrack cần image size
            if len(det_array) > 0:
                max_x = int(det_array[:, 2].max()) + 100
                max_y = int(det_array[:, 3].max()) + 100
            else:
                max_x, max_y = 1920, 1080
            
            online_targets = self.tracker.update(
                det_array,
                [max_y, max_x],
                [max_y, max_x]
            )
        
        # Chuyển đổi sang format dễ dùng
        tracked_objects = []
        current_track_ids = set()
        
        for track in online_targets:
            if not track.is_activated:
                continue
            
            track_id = track.track_id
            current_track_ids.add(track_id)
            
            # Lấy bbox
            tlwh = track.tlwh  # Top-left-width-height
            x1, y1, w, h = tlwh
            x2 = x1 + w
            y2 = y1 + h
            bbox = (int(x1), int(y1), int(x2), int(y2))
            
            # Lưu lịch sử
            if track_id not in self.track_history:
                self.track_history[track_id] = []
                self.total_tracks_created += 1
                # Sự kiện mới: người vào
                self.entry_exit_events.append({
                    'type': 'entry',
                    'track_id': track_id,
                    'frame_id': self.tracker.frame_id,
                    'bbox': bbox
                })
            
            self.track_history[track_id].append(bbox)
            
            # Giữ tối đa 30 frames trong history
            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id].pop(0)
            
            tracked_obj = {
                'track_id': track_id,
                'bbox': bbox,
                'confidence': track.score,
                'tlwh': tlwh,
                'is_activated': track.is_activated,
                'tracklet_len': track.tracklet_len,
                'history': self.track_history[track_id].copy()
            }
            
            tracked_objects.append(tracked_obj)
        
        # Phát hiện tracks bị mất (người ra khỏi lớp)
        previous_ids = set(self.track_history.keys())
        lost_ids = previous_ids - current_track_ids
        
        for lost_id in lost_ids:
            if lost_id not in [e['track_id'] for e in self.entry_exit_events if e['type'] == 'exit']:
                # Chỉ log exit nếu chưa log trước đó
                self.entry_exit_events.append({
                    'type': 'exit',
                    'track_id': lost_id,
                    'frame_id': self.tracker.frame_id,
                    'bbox': self.track_history[lost_id][-1] if self.track_history[lost_id] else None
                })
        
        self.current_tracks = tracked_objects
        return tracked_objects
    
    def get_stable_count(self) -> int:
        """
        Số người ổn định (đang được track)
        """
        return len(self.current_tracks)
    
    def get_entry_exit_summary(self) -> Dict[str, int]:
        """
        Tổng kết số người vào/ra
        """
        entries = sum(1 for e in self.entry_exit_events if e['type'] == 'entry')
        exits = sum(1 for e in self.entry_exit_events if e['type'] == 'exit')
        return {
            'total_entries': entries,
            'total_exits': exits,
            'current_count': self.get_stable_count(),
            'total_tracks_created': self.total_tracks_created
        }
    
    def reset(self):
        """Reset tracker"""
        self.tracker = BYTETracker(
            track_thresh=self.config.track_thresh,
            track_buffer=self.config.track_buffer,
            match_thresh=self.config.match_thresh,
            frame_rate=self.config.frame_rate
        )
        self.track_history.clear()
        self.current_tracks.clear()
        self.entry_exit_events.clear()
        self.total_tracks_created = 0
        logger.info("ByteTracker reset")
