"""
Demo script for Unified Multi-Task Detector
Shows person detection, face detection with landmarks, and hand keypoint detection
All in a single forward pass!
"""

import cv2
import numpy as np
import time
import logging
from pathlib import Path

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
    """Draw face bounding box and landmarks"""
    x1, y1, x2, y2 = face.bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    
    # Draw landmarks
    if face.landmarks is not None:
        for (lx, ly) in face.landmarks:
            cv2.circle(image, (int(lx), int(ly)), 2, (0, 255, 255), -1)
    
    # Draw label
    label = f"Face {face.confidence:.2f}"
    if face.track_id is not None:
        label = f"Face#{face.track_id} {face.confidence:.2f}"
    
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
    cv2.putText(image, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def draw_hand(image: np.ndarray, hand, color=(0, 0, 255)):
    """Draw hand bounding box and keypoints"""
    x1, y1, x2, y2 = hand.bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    
    # Draw keypoints
    if hand.keypoints is not None:
        # Hand connections (MediaPipe hand model)
        connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index finger
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle finger
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring finger
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]
        
        # Draw connections
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx < len(hand.keypoints) and end_idx < len(hand.keypoints):
                start_point = tuple(hand.keypoints[start_idx].astype(int))
                end_point = tuple(hand.keypoints[end_idx].astype(int))
                cv2.line(image, start_point, end_point, (255, 100, 0), 1)
        
        # Draw keypoints
        for i, (kx, ky) in enumerate(hand.keypoints):
            visibility = hand.visibility[i] if hand.visibility is not None else 1.0
            color_intensity = int(255 * visibility)
            cv2.circle(image, (int(kx), int(ky)), 3, (0, color_intensity, 255), -1)
    
    # Draw label
    label = f"Hand {hand.confidence:.2f}"
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image, (x1, y2), (x1 + text_w, y2 + text_h + 4), color, -1)
    cv2.putText(image, label, (x1, y2 + text_h + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def draw_stats(image: np.ndarray, people_count: int, faces_count: int, hands_count: int, 
               inference_time: float, fps: float):
    """Draw statistics overlay"""
    h, w = image.shape[:2]
    
    # Create semi-transparent overlay
    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (300, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    
    # Draw stats
    y_offset = 35
    stats = [
        f"People: {people_count}",
        f"Faces: {faces_count}",
        f"Hands: {hands_count}",
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
    confidence_threshold: float = 0.5
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
        image_size=640
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
            people, faces, hands, inference_time = detector.detect(frame)
            
            # Draw all detections
            vis_frame = frame.copy()
            
            for person in people:
                draw_person(vis_frame, person, color=(0, 255, 0))
            
            for face in faces:
                draw_face(vis_frame, face, color=(255, 0, 0))
            
            for hand in hands:
                draw_hand(vis_frame, hand, color=(0, 0, 255))
            
            # Calculate FPS
            frame_time = time.time() - start_time
            frame_times.append(frame_time)
            if len(frame_times) > 30:
                frame_times.pop(0)
            avg_fps = 1.0 / (sum(frame_times) / len(frame_times))
            
            # Draw stats
            draw_stats(vis_frame, len(people), len(faces), len(hands), 
                      inference_time, avg_fps)
            
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
                          f"{len(faces)} faces, {len(hands)} hands | "
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
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Unified Multi-Task Detector Demo")
    logger.info("=" * 60)
    logger.info(f"Source: {args.source}")
    logger.info(f"Confidence threshold: {args.conf_threshold}")
    
    process_and_display_video(
        source=args.source,
        output_path=args.output,
        display=not args.no_display,
        confidence_threshold=args.conf_threshold
    )


if __name__ == '__main__':
    main()
