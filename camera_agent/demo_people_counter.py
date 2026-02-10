"""
Demo đơn giản để test People Counter module với video file
"""

import cv2
import sys
import os
import time
import numpy as np
import logging
from typing import List, Tuple

# Setup logging
logger = logging.getLogger(__name__)

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from people_counter import create_people_counter, process_video_file

def draw_info(frame: np.ndarray, count: int, boxes: List[Tuple[int, int, int, int, float]], 
              frame_number: int, total_frames: int, fps: float, tracked_objects=None) -> np.ndarray:
    """Vẽ thông tin lên frame"""
    frame_copy = frame.copy()
    
    # Vẽ tracked objects nếu có (ưu tiên)
    # LƯU Ý: Ở đây chỉ đếm người, KHÔNG vẽ trail vì không biết ai là giáo viên
    if tracked_objects is not None and len(tracked_objects) > 0:
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj['bbox']
            track_id = obj['track_id']
            conf = obj['confidence']
            
            # Màu xanh lá đồng nhất cho tất cả (vì không phân biệt được giáo viên/học sinh)
            color = (0, 255, 0)
            
            # Vẽ box đơn giản
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            
            # KHÔNG vẽ track history ở đây vì không biết ai là giáo viên
            # Track history chỉ nên vẽ trong demo_teacher_recognizer_v2.py
            
            # Label đơn giản
            label = f"Person ({conf:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame_copy, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame_copy, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    else:
        # Vẽ bounding boxes thông thường (không có tracking)
        for x1, y1, x2, y2, conf in boxes:
            # Box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Label
            label = f"Person {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame_copy, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(frame_copy, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # Thông tin tổng quan
    info_text = [
        f"People Count: {count}",
        f"Tracked: {len(tracked_objects) if tracked_objects else 0}",
        f"Frame: {frame_number}/{total_frames}",
        f"Progress: {(frame_number/total_frames*100):.1f}%",
        f"FPS: {fps:.1f}"
    ]
    
    y = 30
    for text in info_text:
        cv2.putText(frame_copy, text, (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y += 30
    
    return frame_copy

def process_and_display_video(video_path: str):
    """Xử lý và hiển thị video với people counter"""
    print(f"Processing video: {video_path}")
    counter = create_people_counter(
        confidence_threshold=0.45,
        update_interval=10.0,
        enhance_image=True
    )
    
    # Kích hoạt tracking
    if hasattr(counter.config, 'use_tracking'):
        counter.config.use_tracking = True
        logger.info("Tracking enabled")

    
    if not counter.start():
        print("Failed to start people counter")
        return
    
    # Mở video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video file")
        counter.cleanup()
        return
    
    # Video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video info:")
    print(f"   Total frames: {total_frames}")
    print(f"   FPS: {fps:.1f}")
    
    print("\nControls:")
    print("   'q' - Quit")
    print("   's' - Save count log")
    print("   'SPACE' - Pause/Resume")
    
    frame_count = 0
    paused = False
    start_time = time.time()
    processing_fps = 0
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("Video finished")
                    break
                
                frame_count += 1
                
                # Xử lý frame (với tracking)
                count, boxes, tracked_objects = counter.process_frame(frame)
                
                # Tính FPS
                if frame_count % 30 == 0:
                    current_time = time.time()
                    processing_fps = 30 / (current_time - start_time)
                    start_time = current_time
                
                # Vẽ thông tin (bao gồm tracking)
                display_frame = draw_info(frame, count, boxes, 
                                       frame_count, total_frames,
                                       processing_fps, tracked_objects)
            
            # Hiển thị frame
            cv2.imshow("People Counter Demo", display_frame)
            
            # Progress update
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
            
            # Xử lý phím
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('s'):
                filename = f"count_log_{int(time.time())}.json"
                counter.save_count_log(filename)
                print(f"Count log saved to {filename}")
            elif key == ord(' '):
                paused = not paused
                print("Paused" if paused else "Resumed")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        counter.cleanup()
        
        # Show final stats
        history = counter.get_count_history()
        print(f"\nFinal Statistics:")
        print(f"   Frames processed: {frame_count}/{total_frames}")
        print(f"   Count updates: {len(history)}")
        
        # Tracking summary
        tracking_summary = counter.get_tracking_summary()
        if tracking_summary.get('tracking_enabled'):
            print(f"\nTracking Summary:")
            print(f"   Total entries: {tracking_summary.get('total_entries', 0)}")
            print(f"   Total exits: {tracking_summary.get('total_exits', 0)}")
            print(f"   Tracks created: {tracking_summary.get('total_tracks_created', 0)}")
        
        if history:
            print(f"\n   Latest counts:")
            for record in history[-3:]:
                print(f"     {record['time_str']}: {record['count']} people")

def main():
    """Main demo function"""
    print("People Counter Demo")
    print("=" * 40)
    
    # Tìm video files
    video_paths = [
        "Short_Video_Action/20250915132500_20250915133059_TV_UTT(admin)/Ch05_CH 05_1513_1.avi",
        "Short_Video_Action/20250915133500_20250915134059_TV_UTT(admin)/Ch05_CH 05_1513.avi",
        "Short_Video_Action/20250915135100_20250915135659_TV_UTT(admin)/Ch05_CH 05_1513_3.avi"
    ]
    
    # Liệt kê videos có sẵn
    print("\nAvailable videos:")
    available_videos = []
    for i, path in enumerate(video_paths, 1):
        if os.path.exists(path):
            available_videos.append(path)
            print(f"{i}. {os.path.basename(path)}")
    
    if not available_videos:
        print("No video files found!")
        return
    
    # Chọn video
    while True:
        try:
            choice = int(input("\nChoose a video (1-%d): " % len(available_videos)))
            if 1 <= choice <= len(available_videos):
                video_path = available_videos[choice-1]
                break
            else:
                print("Invalid choice, try again")
        except ValueError:
            print("Please enter a number")
    
    # Process video
    process_and_display_video(video_path)
    print("\nDemo completed!")

if __name__ == "__main__":
    main()