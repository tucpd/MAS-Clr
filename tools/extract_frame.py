import cv2
import os
from pathlib import Path


def extract_frames_every_half_second(video_path: str, output_dir: str) -> None:
    """
    Extract frame từ video với tần suất 30 frame / 1 lần.
    
    Args:
        video_path (str): Đường dẫn tới file video (mp4, avi, mov, ...)
        output_dir (str): Thư mục sẽ lưu các frame (sẽ tự tạo nếu chưa có)
    """
    # Kiểm tra video có tồn tại không
    if not os.path.isfile(video_path):
        print(f"Không tìm thấy file video: {video_path}")
        return
    
    # Tạo thư mục đầu ra nếu chưa tồn tại
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Lấy tên file video (không có phần mở rộng)
    video_name = Path(video_path).stem
    
    # Mở video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Không thể mở video: {video_path}")
        return
    
    # Lấy FPS của video
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("Không lấy được FPS của video, dùng mặc định 30")
        fps = 25.0
    
    # Số frame cần bỏ qua giữa 2 lần chụp
    frame_interval = fps*3
    
    print(f"Video: {video_path}")
    print(f"FPS: {fps:.2f} → Chụp mỗi {frame_interval} frame")
    
    frame_count = 0      # đếm frame thực tế trong video
    saved_count = 0      # đếm số frame đã lưu
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            # Tạo tên file: ten_video_0001.jpg, ten_video_0002.jpg,...
            filename = f"Ch06_CH 06_617_{saved_count+1:04d}_back.jpg"
            save_path = os.path.join(output_dir, filename)
            
            # Lưu frame
            cv2.imwrite(save_path, frame)
            saved_count += 1
            
            # In tiến trình (có thể comment nếu không cần)
            if saved_count % 50 == 0:
                print(f"Đã lưu {saved_count} frame...")
        
        frame_count += 1
    
    cap.release()
    print(f"\nHoàn tất! Đã lưu {saved_count} frame vào thư mục:")
    print(f"→ {output_dir}")
    print(f"Tổng số frame đọc được: {frame_count}")

if __name__ == "__main__":

    video_file = r"/home/chuphamdinhtu/MAS-Clr/data/videos/Ch06_CH 06_617.avi"
    output_folder = r"/home/chuphamdinhtu/MAS-Clr/data/images/Ch06_CH 06_617_back"
    
    extract_frames_every_half_second(video_file, output_folder)