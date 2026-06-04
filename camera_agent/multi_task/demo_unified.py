"""
Demo script for Unified Multi-Task Detector
Shows person detection and face detection
All in a single forward pass!
"""

import cv2
import numpy as np
import time
import logging
from typing import Optional

from unified_detector import UnifiedDetector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def draw_person(image: np.ndarray, person, color=(0, 255, 0)):
    """Draw person bounding box"""
    x1, y1, x2, y2 = person.bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    
    # Draw label
    label = f"Person {person.confidence:.2f}"
    if person.track_id is not None:
        label = f"Person#{person.track_id} {person.confidence:.2f}"
    
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
    cv2.putText(image, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def draw_face(image: np.ndarray, face, color=(255, 0, 0)):
    """Draw face bounding box"""
    x1, y1, x2, y2 = face.bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    
    # Draw label
    label = f"Face {face.confidence:.2f}"
    if face.track_id is not None:
        label = f"Face#{face.track_id} {face.confidence:.2f}"
    
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
    cv2.putText(image, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def draw_stats(image: np.ndarray, people_count: int, faces_count: int,
               inference_time: float, fps: float,
               teacher_id: Optional[str] = None,
               teacher_confidence: Optional[float] = None):
    """Draw statistics overlay"""
    # Create semi-transparent overlay
    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (360, 165), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    
    # Draw stats
    y_offset = 35
    teacher_text = 'Teacher: unknown'
    if teacher_id is not None:
        if teacher_confidence is not None:
            teacher_text = f"Teacher: {teacher_id} ({teacher_confidence:.2f})"
        else:
            teacher_text = f"Teacher: {teacher_id}"

    stats = [
        f"People: {people_count}",
        f"Faces: {faces_count}",
        teacher_text,
        f"Inference: {inference_time:.1f} ms",
        f"FPS: {fps:.1f}"
    ]
    
    for stat in stats:
        cv2.putText(image, stat, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (0, 255, 0), 1, cv2.LINE_AA)
        y_offset += 25


def process_and_display_video(
    source: str = '0',
    output_path: str = None,
    display: bool = True,
    confidence_threshold: float = 0.5,
    enable_teacher_recognition: bool = False,
    teacher_db_path: Optional[str] = None,
    teacher_preferences_path: Optional[str] = None,
):
    """
    Process video with unified detector
    
    Args:
        source: Video source (file path, webcam index, or RTSP URL)
        output_path: Path to save output video (optional)
        display: Show real-time display window
        confidence_threshold: Detection confidence threshold
    """
    # Create detector
    logger.info(f"Initializing unified detector...")
    detector = UnifiedDetector(
        backbone_name='mobilenetv3_large_100',
        confidence_threshold=confidence_threshold,
        image_size=640,
        enable_teacher_recognition=enable_teacher_recognition,
        teacher_db_path=teacher_db_path,
        teacher_preferences_path=teacher_preferences_path,
    )
    
    # Open video source
    if source.isdigit():
        source = int(source)
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Failed to open video source: {source}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Video: {width}x{height} @ {fps} FPS")
    
    # Setup video writer
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        logger.info(f"Saving output to: {output_path}")
    
    # FPS calculation
    frame_times = []
    frame_count = 0
    
    logger.info("Processing video... Press 'q' to quit, 's' to save screenshot")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video")
                break
            
            frame_count += 1
            start_time = time.time()
            
            # Run unified detection
            if enable_teacher_recognition:
                people, faces, camera_output, inference_time = detector.detect_with_output(frame)
                teacher_id = camera_output.teacher_id
                teacher_confidence = camera_output.teacher_confidence
            else:
                people, faces, inference_time = detector.detect(frame)
                teacher_id = None
                teacher_confidence = None
            
            # Draw all detections
            vis_frame = frame.copy()
            
            for person in people:
                draw_person(vis_frame, person, color=(0, 255, 0))
            
            for face in faces:
                draw_face(vis_frame, face, color=(255, 0, 0))
            
            # Calculate FPS
            frame_time = time.time() - start_time
            frame_times.append(frame_time)
            if len(frame_times) > 30:
                frame_times.pop(0)
            avg_fps = 1.0 / (sum(frame_times) / len(frame_times))
            
            # Draw stats
            draw_stats(vis_frame, len(people), len(faces),
                      inference_time, avg_fps,
                      teacher_id=teacher_id,
                      teacher_confidence=teacher_confidence)
            
            # Write to output
            if writer:
                writer.write(vis_frame)
            
            # Display
            if display:
                cv2.imshow('Unified Multi-Task Detector', vis_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("User quit")
                    break
                elif key == ord('s'):
                    screenshot_path = f"screenshot_{frame_count}.jpg"
                    cv2.imwrite(screenshot_path, vis_frame)
                    logger.info(f"Screenshot saved: {screenshot_path}")
            
            # Log progress
            if frame_count % 30 == 0:
                logger.info(f"Frame {frame_count}: {len(people)} people, "
                          f"{len(faces)} faces, teacher={teacher_id} | "
                          f"{inference_time:.1f}ms | {avg_fps:.1f} FPS")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()
        
        logger.info(f"Processed {frame_count} frames")
        if frame_times:
            avg_fps = 1.0 / (sum(frame_times) / len(frame_times))
            logger.info(f"Average FPS: {avg_fps:.1f}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Unified Multi-Task Detector Demo')
    parser.add_argument('--source', type=str, default='0',
                       help='Video source (webcam index, video file, or RTSP URL)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video path (optional)')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable display window')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                       help='Detection confidence threshold')
    parser.add_argument('--enable-teacher-recognition', action='store_true',
                       help='Enable ArcFace-based teacher recognition from face bboxes')
    parser.add_argument('--teacher-db', type=str, default=None,
                       help='Path to ArcFace teacher embedding DB (pkl)')
    parser.add_argument('--teacher-preferences', type=str, default=None,
                       help='Path to teacher preferences DB (json/pkl)')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Unified Multi-Task Detector Demo")
    logger.info("=" * 60)
    logger.info(f"Source: {args.source}")
    logger.info(f"Confidence threshold: {args.conf_threshold}")
    logger.info(f"Teacher recognition: {'on' if args.enable_teacher_recognition else 'off'}")
    
    process_and_display_video(
        source=args.source,
        output_path=args.output,
        display=not args.no_display,
        confidence_threshold=args.conf_threshold,
        enable_teacher_recognition=args.enable_teacher_recognition,
        teacher_db_path=args.teacher_db,
        teacher_preferences_path=args.teacher_preferences,
    )


if __name__ == '__main__':
    main()
