"""
Demo Teacher Recognition V2
Demonstrates teacher recognition with combined face recognition and person tracking
"""

import cv2
import sys
import os
import time
import numpy as np
from typing import List, Dict, Any

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from teacher_recognizer_v2 import TeacherRecognizer, TeacherRecognizerConfig

def draw_detections(frame: np.ndarray, 
                   detections: List[Dict[str, Any]],
                   frame_info: Dict[str, Any]) -> np.ndarray:
    """
    Vẽ thông tin nhận diện lên frame
    
    Args:
        frame: Input frame
        detections: List of person detections with tracking info
        frame_info: Additional frame information
        
    Returns:
        Annotated frame
    """
    frame_copy = frame.copy()
    
    # Vẽ thông tin xử lý
    info_text = [
        f"Frame: {frame_info['frame_count']}",
        f"FPS: {frame_info['fps']:.1f}",
        f"Tracking: {len(detections)} people",
        f"Teachers: {frame_info['teacher_count']}"
    ]
    
    y = 30
    for text in info_text:
        cv2.putText(frame_copy, text, (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y += 30
    
    return frame_copy

def process_video_file(video_path: str, recognizer: TeacherRecognizer):
    """
    Xử lý video file với teacher recognition V2
    
    Args:
        video_path: Đường dẫn tới video file
        recognizer: TeacherRecognizer instance
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
    print("   'd' - Show database info")
    print("   't' - Show tracking info")
    
    frame_count = 0
    paused = False
    start_time = time.time()
    processing_fps = 0
    show_tracks = False
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("Video finished")
                    break
                
                frame_count += 1
                
                # Process frame
                detections, annotated_frame = recognizer.process_frame(frame)
                
                # Calculate FPS
                if frame_count % 30 == 0:
                    current_time = time.time()
                    processing_fps = 30 / (current_time - start_time)
                    start_time = current_time
                
                # Count recognized teachers
                teacher_count = len(set(d["teacher_id"] for d in detections if d["teacher_id"]))
                
                # Frame info for visualization
                frame_info = {
                    "frame_count": frame_count,
                    "fps": processing_fps,
                    "teacher_count": teacher_count
                }
                
                # Add info overlay
                display_frame = draw_detections(annotated_frame, detections, frame_info)
                
                # Add progress bar
                progress = (frame_count / total_frames) * 100
                cv2.putText(display_frame, f"Progress: {progress:.1f}%",
                           (10, display_frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow("Teacher Recognition V2 Demo", display_frame)
            
            # Progress update
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
            
            # Handle keys
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('p'):
                paused = not paused
                print("Paused" if paused else "Resumed")
            elif key == ord('d'):
                info = recognizer.get_database_info()
                print("\nDatabase Info:")
                print(f"Total teachers: {info['total_teachers']}")
                if info['total_teachers'] > 0:
                    print("Teachers:")
                    for teacher_id, count in info['embeddings_per_teacher'].items():
                        print(f"   {teacher_id}: {count} embeddings")
                print()
            elif key == ord('t'):
                show_tracks = not show_tracks
                if show_tracks:
                    print("\nActive Tracks:")
                    for track_id, track in recognizer.tracks.items():
                        status = "Teacher" if track.teacher_id else "Unknown"
                        print(f"Track #{track_id}: {status}")
                        if track.teacher_id:
                            print(f"   Teacher: {track.teacher_id}")
                            print(f"   Confidence: {track.confidence:.2f}")
                        print(f"   Missing frames: {track.consecutive_misses}")
                print()
    
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
    print("Teacher Recognition V2 Demo")
    print("=" * 40)
    
    # Khởi tạo recognizer với database có sẵn
    config = TeacherRecognizerConfig(
        database_path="teacher_database.pkl",  # Sử dụng database có sẵn
        confidence_threshold=0.9,  # Face recognition confidence (FaceNet)
        arcface_threshold=0.5,  # Face recognition threshold (ArcFace)
        face_confidence=0.5,  # Face detection confidence
        person_confidence=0.5,  # Person detection confidence
        tracking_timeout=3.0,  # Seconds until track is lost
        recognition_interval=1.0,  # Recognize every second
        use_arcface=True,  # Dùng ArcFace (nhanh hơn, chính xác hơn)
        arcface_model="buffalo_sc"  # Small model (16MB)
    )
    
    recognizer = TeacherRecognizer(config)
    
    if not recognizer.initialize():
        print("Failed to initialize recognizer")
        return
    
    # Hiển thị thông tin database
    info = recognizer.get_database_info()
    print("\nLoaded Teacher Database Info:")
    print(f"Total teachers: {info['total_teachers']}")
    if info['total_teachers'] > 0:
        print("Teachers:")
        for teacher_id, count in info['embeddings_per_teacher'].items():
            print(f"   {teacher_id}: {count} embeddings")
    
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
    process_video_file(video_path, recognizer)
    
    # Cleanup
    recognizer.cleanup()
    print("\nDemo completed")

if __name__ == "__main__":
    main()