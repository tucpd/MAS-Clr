# Bao Cao Ky Thuat: Camera Agent (Person + Face + Teacher Recognition)

## 1. Tong quan
Camera Agent hien tai duoc toi gian theo huong:
- Detect nguoi (person bbox)
- Detect khuon mat (face bbox)
- Nhan dien giao vien tu face bbox bang ArcFace (optional)

Muc tieu la giam do phuc tap, giam tai tinh toan, va phu hop bai toan lop hoc thong minh thoi gian thuc.

## 2. Kien truc hien tai
Pipeline xu ly trong nhanh multi_task:

1. Frame input (BGR)
2. Preprocess: letterbox ve 640x640 + ImageNet normalization
3. Backbone dung chung: MobileNetV3-Large + FPN
4. Heads:
- PersonHead: person_cls, person_reg, person_ctr
- FaceHead: face_cls, face_reg, face_ctr
5. Postprocess:
- FCOS decode (anchor-free)
- NMS cho person, face
6. Teacher recognition (optional):
- Lay face bbox da detect
- ArcFace detect/extract trong ROI
- Cosine similarity voi teacher DB
- Sinh output JSON cho control side

## 3. File chinh
- camera_agent/multi_task/backbone.py
- camera_agent/multi_task/heads.py
- camera_agent/multi_task/unified_model.py
- camera_agent/multi_task/unified_detector.py
- camera_agent/multi_task/losses.py
- camera_agent/multi_task/demo_unified.py

## 4. Chi tiet mo hinh
### 4.1 Shared Backbone
- Backbone: mobilenetv3_large_100 (timm)
- Out indices: [2, 3, 4]
- FPN outputs:
- p3 (stride 8)
- p4 (stride 16)
- p5 (stride 32)

### 4.2 PersonHead
- Chay tren p3, p4, p5
- Output moi level:
- person_cls: [B, 1, H, W]
- person_reg: [B, 4, H, W]
- person_ctr: [B, 1, H, W]

### 4.3 FaceHead
- Chay tren p3, p4
- Output moi level:
- face_cls: [B, 1, H, W]
- face_reg: [B, 4, H, W]
- face_ctr: [B, 1, H, W]

Luu y: da bo hoan toan face landmarks, hand, keypoints.

## 5. Teacher recognition (face bbox only)
Teacher recognition duoc trien khai trong UnifiedDetector theo kieu optional:
- Co the bat/tat bang co enable_teacher_recognition
- Khong bat se van detect person/face binh thuong

### 5.1 Luong xu ly
1. Sap xep face detections theo confidence
2. Thu toi da 3 face bbox co confidence cao nhat
3. Goi ArcFaceRecognizer.detect_and_extract voi roi=(x1, y1, x2, y2)
4. Nhan embedding, so khop qua ArcFaceRecognizer.recognize_face
5. Neu match teacher_id:
- Tra ve teacher_id
- teacher_confidence (cosine similarity)
- teacher_preferences (neu co)

### 5.2 Du lieu giao vien
- Embedding DB: pkl (duong dan truyen qua teacher_db_path)
- Preferences DB: json/pkl (teacher_preferences_path)

Preferences la optional, neu khong co se tra ve dict rong {}.

## 6. API output
### 6.1 Detection API co ban
- detect(image) -> (people, faces, inference_time_ms)

### 6.2 Detection + output cho control agent
- detect_with_output(image) -> (people, faces, camera_output, inference_time_ms)

Trong do camera_output la dataclass CameraAgentOutput:
- num_students: int
- teacher_id: Optional[str]
- teacher_confidence: Optional[float]
- teacher_preferences: Dict[str, Any]
- gesture_command: Optional[str] (hien de None de giu tuong thich)

### 6.3 JSON output
CameraAgentOutput.to_json() tra ve JSON co cac truong:
- num_students
- teacher_id
- teacher_confidence
- teacher_preferences
- gesture_command

## 7. Loss hien tai
Trong losses.py, MultiTaskLoss da duoc rut gon theo task moi:
- person_cls, person_reg, person_ctr
- face_cls, face_reg, face_ctr

Da bo cac task:
- face_lmk
- hand_cls, hand_reg, hand_ctr
- hand_kpt, hand_hm

## 8. Demo
demo_unified.py da cap nhat:
- Hien thi person + face
- Khong ve hand/keypoint/landmark
- Ho tro co --enable-teacher-recognition
- Co --teacher-db va --teacher-preferences de nap DB

Vi du:

```bash
python -m camera_agent.multi_task.demo_unified \
  --source 0 \
  --enable-teacher-recognition \
  --teacher-db camera_agent/teacher_database.pkl \
  --teacher-preferences camera_agent/teacher_preferences.example.json
```

## 9. Phu thuoc
requirements.txt da duoc tach theo pipeline hien tai:
- Core: opencv, numpy, torch, torchvision, timm
- Teacher recognition: insightface, onnxruntime-gpu
- Legacy old_architecture duoc chuyen sang optional (comment)

## 10. Gioi han hien tai
1. Training pipeline cho nhanh multi_task chua duoc dong goi thanh bo trainer/evaluator day du.
2. Teacher preferences can duoc cung cap boi file json/pkl rieng neu muon output co gia tri.
3. Teacher recognition la optional, phu thuoc su san sang cua insightface va DB embedding.

## 11. Huong tiep theo de san sang production
1. Chot schema output cuoi cung voi control agent (giu hay bo gesture_command).
2. Dong bo event-driven output theo SRS (chi gui khi thay doi lon).
3. Them bo test video regression cho:
- teacher vao/ra nhanh
- che mat tam thoi
- nhieu khuon mat cung luc
4. Chot KPI teacher recognition:
- false positive
- false negative
- do on dinh teacher_id theo thoi gian
