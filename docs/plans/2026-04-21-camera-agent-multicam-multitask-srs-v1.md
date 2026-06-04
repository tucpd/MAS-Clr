# Camera Agent Multi-Camera Multi-Task - SRS v1

## 1. Mục tiêu
Tài liệu này đặc tả yêu cầu cho Camera Agent trong hệ thống lớp học thông minh, với mục tiêu:
- Xử lý đồng thời 2 camera (trước, sau) có timestamp đồng bộ.
- Thực hiện đa tác vụ theo thời gian thực: đếm người, nhận diện khuôn mặt, xác định giáo viên, theo dõi giáo viên xuyên camera.
- Xuất dữ liệu sự kiện cho Control Agent với độ trễ tối đa 5 giây.

## 2. Phạm vi
- Bao gồm logic xử lý video và hợp nhất kết quả liên camera tại Camera Agent.
- Không bao gồm logic điều khiển thiết bị ở Control Agent.
- Không bao gồm quy trình huấn luyện chi tiết mô hình (chỉ định nghĩa yêu cầu đầu ra và tiêu chí chấp nhận).

## 3. Bối cảnh và giả định
- Mỗi ca học thường có 1 giáo viên chính.
- Có thể xuất hiện thêm 1 giáo viên khác đi vào lớp trong thời gian ngắn rồi đi ra.
- 2 camera cố định góc nhìn, đã chia vùng quan sát để không chồng lấn.
- FPS mỗi camera: 25.
- 2 camera đồng bộ timestamp.
- Tổng số người trong lớp = tổng số người của 2 vùng camera sau khi áp dụng quy tắc vùng biên chống dao động.

## 4. Đầu vào
### 4.1 Luồng video
- `camera_front`: luồng video 25 FPS.
- `camera_back`: luồng video 25 FPS.
- Mỗi frame có `timestamp` đồng bộ để ghép cặp theo thời gian.

### 4.2 Dữ liệu nhận diện giáo viên
- Vector database chứa embedding khuôn mặt giáo viên đã đăng ký.
- Mỗi giáo viên có `teacher_id` duy nhất.

### 4.3 Cấu hình hệ thống
- ROI/polygon hợp lệ cho từng camera.
- Đường biên vùng và vùng đệm biên (hysteresis band).
- Các ngưỡng xác nhận và duy trì danh tính.

## 5. Đầu ra
Camera Agent xuất event JSON cho Control Agent khi có thay đổi lớn.

Ví dụ cấu trúc event:

```json
{
  "event_id": "uuid",
  "timestamp": "2026-04-21T09:30:01.120Z",
  "type": "teacher_state_changed",
  "classroom_id": "CH05",
  "summary": {
    "total_people_count": 37,
    "camera_front_count": 19,
    "camera_back_count": 18,
    "teacher": {
      "teacher_id": "T001",
      "state": "confirmed",
      "owner_camera": "front",
      "confidence": 0.91,
      "track_id": "front_128"
    }
  },
  "detail": {
    "reason": "reacquired_after_short_occlusion",
    "changed_fields": ["teacher.state", "teacher.owner_camera"]
  }
}
```

## 6. Yêu cầu chức năng
### FR-01: Đồng bộ liên camera
- Hệ thống phải ghép cặp frame 2 camera theo timestamp.
- Sai lệch timestamp cho phép cần được cấu hình (khuyến nghị <= 100 ms).

### FR-02: Đếm người theo từng camera
- Mỗi camera phát hiện và theo dõi người trong ROI của camera đó.
- Điểm neo đếm người sử dụng `điểm chân bbox` (tâm cạnh dưới bbox người).
- Người được tính vào camera khi điểm chân nằm trong ROI hợp lệ.

### FR-03: Quy tắc vùng biên chống dao động
- Phải có vùng đệm biên với độ rộng cấu hình được (khuyến nghị 3-5% bề rộng ảnh).
- Nếu điểm chân nằm trong vùng đệm biên, giữ nguyên camera/vùng đã gán trước đó.
- Chỉ đổi vùng khi đối tượng ở vùng mới liên tục `M` frame (khuyến nghị M = 10).
- Cập nhật people_count chỉ khi ổn định `K` frame (khuyến nghị K = 8-12).

### FR-04: Nhận diện và xác nhận giáo viên
- So khớp embedding khuôn mặt phát hiện được với vector database.
- Xác nhận giáo viên chỉ khi đạt điều kiện ổn định `N` frame liên tiếp (khuyến nghị N = 10).
- Hệ thống ưu tiên giảm false positive.

### FR-05: Duy trì danh tính giáo viên khi mất mặt tạm thời
- Khi không thấy mặt giáo viên, hệ thống giữ danh tính tạm thời tối đa 60 giây.
- Trong thời gian giữ tạm, hệ thống vẫn duy trì `teacher_id` nếu track còn hợp lệ theo tracker.

### FR-06: Theo dõi giáo viên xuyên camera
- Giáo viên phải giữ nguyên một `teacher_id` xuyên suốt giữa 2 camera.
- Nếu cả 2 camera cùng thấy giáo viên ở vùng biên cùng thời điểm, ưu tiên camera đang là owner trước đó.
- Chỉ chuyển owner camera khi owner hiện tại suy giảm tin cậy liên tục `L` frame và camera còn lại ổn định đủ điều kiện (khuyến nghị L = 10).

### FR-07: Xử lý giáo viên phụ xuất hiện ngắn
- Mặc định không thay giáo viên chính trong ca học khi có giáo viên khác xuất hiện ngắn.
- Giáo viên phụ được gắn nhãn `secondary_teacher_candidate` để theo dõi nhưng không takeover ngay.
- Điều kiện takeover (nếu bật) phải cấu hình tường minh theo thời lượng hiện diện.

### FR-08: Event-driven output
- Chỉ gửi output khi có thay đổi lớn, bao gồm:
  - people_count thay đổi ổn định.
  - teacher_state thay đổi (`confirmed`, `temporarily_lost`, `reacquired`, `absent_timeout`).
  - owner_camera thay đổi.
  - teacher_id thay đổi (mức cảnh báo cao).
- Có cơ chế chống bắn event quá dày (debounce, khuyến nghị >= 1 giây/event type).

## 7. Yêu cầu phi chức năng
### NFR-01: Độ trễ
- Độ trễ end-to-end từ frame ingest đến event publish phải <= 5 giây (P95).

### NFR-02: Độ ổn định
- Hệ thống phải hoạt động liên tục theo phiên học, không reset danh tính giáo viên ngoài các điều kiện timeout.

### NFR-03: Khả năng giám sát
- Mỗi event cần có metadata truy vết: timestamp, camera owner, confidence, reason.
- Hệ thống cần log các trạng thái chuyển tiếp để phục vụ debug.

## 8. Máy trạng thái giáo viên (đề xuất)
- `unconfirmed`: chưa đủ N frame xác nhận.
- `confirmed`: giáo viên chính đã xác nhận.
- `temporarily_lost`: mất mặt/tín hiệu ngắn hạn, còn trong ngưỡng 60 giây.
- `reacquired`: tìm lại được giáo viên trong thời gian giữ tạm.
- `absent_timeout`: quá 60 giây không đủ bằng chứng duy trì.

Chuyển trạng thái chính:
- `unconfirmed -> confirmed`: đủ N frame + vượt ngưỡng xác nhận.
- `confirmed -> temporarily_lost`: mất quan sát ngắn hạn.
- `temporarily_lost -> reacquired -> confirmed`: khôi phục thành công.
- `temporarily_lost -> absent_timeout`: quá ngưỡng giữ tạm.

## 9. Tham số cấu hình khởi tạo
- `fps_front = 25`
- `fps_back = 25`
- `max_end_to_end_latency_sec = 5`
- `teacher_confirm_frames_N = 10`
- `teacher_temp_hold_sec = 60`
- `border_hysteresis_ratio = 0.04`
- `border_switch_frames_M = 10`
- `count_stable_frames_K = 10`
- `owner_handoff_frames_L = 10`
- `event_debounce_sec = 1`

Lưu ý: Các ngưỡng similarity embedding (`T_high`, `T_low`) phải hiệu chỉnh trên tập validation thực tế để ưu tiên giảm false positive.

## 10. Tiêu chí chấp nhận (KPI đề xuất cần chốt)
- P95 latency <= 5 giây.
- Tỷ lệ false positive nhận diện giáo viên ở mức thấp theo mục tiêu vận hành (đề xuất bắt đầu: <= 1 lần/giờ/camera).
- Tỷ lệ handoff giáo viên xuyên camera đúng >= 95% trên tập test có di chuyển qua biên.
- Dao động đếm người tại vùng biên giảm rõ rệt so với baseline không hysteresis.

## 11. Trường hợp biên bắt buộc test
- Giáo viên đi qua lại vùng biên liên tục.
- Giáo viên quay lưng/che mặt ngắn hạn rồi xuất hiện lại.
- Cả 2 camera cùng thấy giáo viên tại biên.
- Giáo viên phụ vào lớp ngắn rồi rời đi.
- Lớp đông người, nhiều che khuất, thay đổi ánh sáng.

## 12. Điểm cần xác nhận thêm trước khi khóa SRS
- Có cho phép tự động takeover giáo viên chính nếu giáo viên phụ hiện diện đủ lâu không? Nếu có, cần chốt ngưỡng thời gian.
- Mục tiêu KPI định lượng chính thức cho false positive nhận diện giáo viên.

## 13. Tai lieu ke hoach implement detector from scratch
- Ke hoach chi tiet module/task/DoD/handoff cho agent moi:
  - `docs/plans/2026-04-24-detector-from-scratch-implementation-plan-v1.md`
