import json
from typing import Dict, Any

# Cấu hình
OPENAI_API_KEY = ""  # 
INPUT_FILE = "input_snapshot.json"  # File input từ Camera Agent
OUTPUT_LOG_FILE = "commands_log.json"  # Log output

# System Prompt (Rules tùy chỉnh - chỉnh theo dự án)
SYSTEM_PROMPT = """
Bạn là Device Control Agent cho lớp học thông minh. Phân tích input và generate JSON commands dựa trên rules sau. 
Output CHỈ JSON hợp lệ theo schema, không text khác. Nếu không cần thay đổi, giữ nguyên status hiện tại.

Rules:
- Đèn (lights_pairs): Nếu lux < 50 hoặc num_students > 15, bật cặp 1 (status=1). Nếu preferences.light_brightness='high', bật cả 2 cặp. Ưu tiên preferences.
- Quạt (fans_pairs): Nếu temperature > 27, set speed=99 (high) cho cặp 1; nếu humidity > 70, speed=50 (med) cho cặp 2. Nếu preferences.fan_speed='auto', điều chỉnh dựa trên temp.
- Điều hòa (air_conditioner): Nếu temperature > 26 và status hiện=0, bật (status=1), set_temp = teacher_preferences.ac_temp (default 24), fan_speed="med".
- Máy chiếu (projector): Bật (1) nếu num_students > 0 và lux > 100; tắt nếu num_students=0.
- Ưu tiên: teacher_preferences > environment > default. Giữ nguyên nếu input ổn định.
"""

# Few-shot example cho prompt (để hướng dẫn LLM)
EXAMPLE_INPUT = {
    "num_students": 25,
    "teacher_preferences": {"ac_temp": 24},
    "environment": {"temperature": 28.5, "humidity": 75, "lux": 40},
    "devices": {"lights_pairs": [{"status": 0}, {"status": 0}], "fans_pairs": [{"speed": 0}, {"speed": 0}], "air_conditioner": {"status": 0, "set_temperature": None, "fan_speed": None}, "projector": 0}
}
EXAMPLE_OUTPUT = {
    "commands": {
        "lights_pairs": [{"status": 1}, {"status": 0}],
        "fans_pairs": [{"speed": 99}, {"speed": 50}],
        "air_conditioner": {"status": 1, "set_temperature": 24.0, "fan_speed": "med"},
        "projector": 1
    },
    "reasoning": "Bật đèn cặp 1 vì lux thấp và đông học sinh; quạt high cặp 1 vì temp cao, med cặp 2 vì humidity cao; bật AC vì temp>26; bật projector vì lớp đông."
}