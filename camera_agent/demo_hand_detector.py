"""
Demo Hand Detection Module - Demo nhận diện bàn tay trong ROI của giáo viên
"""

import cv2
import numpy as np
import sys
import os
import time
from typing import List, Dict, Any

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from teacher_recognizer_v2 import TeacherRecognizer, TeacherRecognizerConfig
from hand_detector import HandDetector

def process_video_file(video_path: str, teacher_recognizer: TeacherRecognizer, 
                      hand_detector: HandDetector):
    """
    Xử lý video file với teacher recognition và hand detection
    
    Args:
        video_path: Đường dẫn tới video file
        teacher_recognizer: TeacherRecognizer instance
        hand_detector: HandDetector instance
    """
    print(f"Processing video: {video_path}")
    
    # Mở video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video file")
        return
    
    # Video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video info:")
    print(f"   Total frames: {total_frames}")
    print(f"   FPS: {fps}")
    
    print("\nControls:")
    print("   'q' - Quit")
    print("   'p' - Pause/Resume")
    print("   'h' - Toggle hand detection")
    print("   'r' - Reset smoothing buffers")
    
    frame_count = 0
    paused = False
    start_time = time.time()
    processing_fps = 0
    detect_hands = True
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("Video finished")
                    break
                
                frame_count += 1
                current_time = time.time()
                
                # Process teacher detection first
                teacher_detections, _ = teacher_recognizer.process_frame(frame)
                
                # Find teacher ROI
                teacher_roi = None
                for det in teacher_detections:
                    if det["teacher_id"]:  # Found a teacher
                        teacher_roi = det["box"]
                        break
                
                if detect_hands and teacher_roi is not None:
                    # Detect hands within teacher ROI (với smoothing)
                    frame, hand_detections = hand_detector.get_stable_hands(frame, teacher_roi)
                    
                    # Draw teacher box
                    x1, y1, x2, y2 = teacher_roi
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "Teacher ROI", (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Hiển thị gesture nếu có
                    if hand_detections:
                        gesture_y = y2 + 30
                        for i, hand_info in enumerate(hand_detections):
                            gesture_text = f"{hand_info['label']}: {hand_info.get('gesture', 'N/A')}"
                            cv2.putText(frame, gesture_text, (x1, gesture_y + i*25),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                else:
                    hand_detections = []
                
                # Calculate FPS
                if frame_count % 30 == 0:
                    processing_fps = 30 / (time.time() - start_time)
                    start_time = time.time()
                
                # Thêm gesture info nếu có
                if hand_detections:
                    gestures = [h.get('gesture', 'N/A') for h in hand_detections]
                    info_text.append(f"Gestures: {', '.join(gestures)}")
                
                # Add info overlay
                info_text = [
                    f"Frame: {frame_count}",
                    f"FPS: {processing_fps:.1f}",
                    f"Hands: {len(hand_detections)}",
                    f"Hand Detection: {'On' if detect_hands else 'Off'}"
                ]
                
                y = 30
                for text in info_text:
                    cv2.putText(frame, text, (10, y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    y += 30
                
                # Add progress bar
                progress = (frame_count / total_frames) * 100
                cv2.putText(frame, f"Progress: {progress:.1f}%",
                           (10, frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow("Teacher Hand Detection Demo", frame)
            
            # Progress update
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
            
            # Handle keys
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('r'):
                hand_detector.reset_buffers()
                print("Smoothing buffers reset")
                paused = not paused
                print("Paused" if paused else "Resumed")
            elif key == ord('h'):
                detect_hands = not detect_hands
                status = "enabled" if detect_hands else "disabled"
                print(f"Hand detection {status}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        print("\nProcessing Summary:")
        print(f"Processed {frame_count}/{total_frames} frames")
        print(f"Average FPS: {frame_count/(time.time()-start_time):.1f}")

def main():
    """Main demo function"""
    print("Teacher Hand Detection Demo")
    print("=" * 40)
    
    # Initialize teacher recognizer
    teacher_config = TeacherRecognizerConfig(
        database_path="teacher_database.pkl",
        confidence_threshold=0.7,
        recognition_interval=1.0
    )
    
    teacher_recognizer = TeacherRecognizer(teacher_config)
    if not teacher_recognizer.initialize():
        print("Failed to initialize teacher recognizer")
        return
        
    # Initialize hand detector với smoothing
    hand_detector = HandDetector(
        static_mode=False,
        max_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        smoothing_window=5  # Smooth qua 5 frames
    )
    
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
    process_video_file(video_path, teacher_recognizer, hand_detector)
    
    # Cleanup
    teacher_recognizer.cleanup()
    hand_detector.cleanup()
    print("\nDemo completed")

if __name__ == "__main__":
    main()