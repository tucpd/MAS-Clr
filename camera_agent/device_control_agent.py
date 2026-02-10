import json
import os
from datetime import datetime
from typing import Dict, Any

import openai
from config import (OPENAI_API_KEY, INPUT_FILE, OUTPUT_LOG_FILE, SYSTEM_PROMPT, EXAMPLE_INPUT, EXAMPLE_OUTPUT)
from models import DeviceCommands

def load_input(file_path: str) -> Dict[str, Any]:
    """Load input JSON từ file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file {file_path} not found. Tạo file mẫu trước.")
    with open(file_path, 'r') as f:
        data = json.load(f)
    # Lấy preferences nếu có (giả sử lưu riêng, nhưng merge vào input)
    data['teacher_preferences'] = data.get('teacher_preferences', {})  # Default empty
    return data

def fallback_commands(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Rule-based fallback nếu LLM fail."""
    env = input_data['environment']
    num_students = input_data.get('num_students', 0)
    prefs = input_data.get('teacher_preferences', {})
    
    # Simple rules (không dùng LLM)
    lights_status = [0, 0]  # Default off
    if env['lux'] < 50 or num_students > 15:
        lights_status[0] = 1  # Bật cặp 1
    
    fans_speed = [0, 0]
    if env['temperature'] > 27:
        fans_speed[0] = 99
    if env['humidity'] > 70:
        fans_speed[1] = 50
    
    ac_status = 1 if env['temperature'] > 26 else 0
    ac_temp = prefs.get('ac_temp', 24.0)
    ac_fan = "med" if ac_status == 1 else None
    
    projector = 1 if num_students > 0 and env['lux'] > 100 else 0
    
    return {
        "commands": {
            "lights_pairs": [{"status": lights_status[0]}, {"status": lights_status[1]}],
            "fans_pairs": [{"speed": fans_speed[0]}, {"speed": fans_speed[1]}],
            "air_conditioner": {"status": ac_status, "set_temperature": ac_temp if ac_status else None, "fan_speed": ac_fan},
            "projector": projector
        },
        "reasoning": "Fallback rule-based: Adjusted based on temp/humidity/lux."
    }

def generate_commands(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate commands bằng LLM."""
    openai.api_key = OPENAI_API_KEY
    
    # Few-shot example: Thêm mẫu input-output để hướng dẫn LLM
    user_prompt = f"""
    Input: {json.dumps(input_data, indent=2)}
    
    Ví dụ: Với input mẫu sau, output mong muốn là:
    Input mẫu: {json.dumps(EXAMPLE_INPUT, indent=2)}
    Output mẫu: {json.dumps(EXAMPLE_OUTPUT, indent=2)}
    
    Generate commands JSON theo schema tương tự ví dụ trên:
    {{
        "commands": {{
            "lights_pairs": [{{"status": 0 or 1}}, {{"status": 0 or 1}}],
            "fans_pairs": [{{"speed": 0 or 25 or 50 or 99}}, {{"speed": 0 or 25 or 50 or 99}}],
            "air_conditioner": {{
                "status": 0 or 1,
                "set_temperature": null or float,
                "fan_speed": null or "low" or "med" or "high"
            }},
            "projector": 0 or 1
        }},
        "reasoning": "Lý do ngắn gọn"
    }}
    """
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.1,  # Low cho consistent
            max_tokens=300
        )
        json_str = response.choices[0].message.content.strip()
        
        # Parse và validate
        commands = DeviceCommands.parse_raw(json_str)
        return {
            "commands": commands.commands,
            "reasoning": commands.reasoning,
            "source": "LLM"
        }
    except Exception as e:
        print(f"LLM Error: {e}. Using fallback.")
        return fallback_commands(input_data)

def apply_commands(commands: Dict[str, Any], current_devices: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate apply commands (thay đổi device status). Trong real, gửi MQTT/API."""
    new_devices = current_devices.copy()
    
    # Áp dụng lights
    for i, pair in enumerate(commands['lights_pairs']):
        new_devices['lights_pairs'][i]['status'] = pair['status']
    
    # Áp dụng fans
    for i, pair in enumerate(commands['fans_pairs']):
        new_devices['fans_pairs'][i]['speed'] = pair['speed']
    
    # Áp dụng AC
    new_devices['air_conditioner'].update(commands['air_conditioner'])
    
    # Áp dụng projector
    new_devices['projector'] = commands['projector']
    
    # Log changes
    changes = {k: v for k, v in new_devices.items() if v != current_devices.get(k)}
    print(f"Applied changes: {json.dumps(changes, indent=2)}")
    
    return new_devices

def log_output(input_data: Dict[str, Any], output: Dict[str, Any], new_devices: Dict[str, Any]):
    """Lưu log vào file JSON (append)."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "input_summary": {k: v for k, v in input_data.items() if k != 'teacher_preferences'},  # Anonymize prefs
        "commands": output,
        "new_device_status": new_devices
    }
    
    if os.path.exists(OUTPUT_LOG_FILE):
        with open(OUTPUT_LOG_FILE, 'r') as f:
            logs = json.load(f)
    else:
        logs = []
    
    logs.append(log_entry)
    with open(OUTPUT_LOG_FILE, 'w') as f:
        json.dump(logs, f, indent=2)
    
    print(f"Logged to {OUTPUT_LOG_FILE}")

def main():
    """Chạy pipeline."""
    try:
        # Bước 1: Load input
        input_data = load_input(INPUT_FILE)
        print("Loaded input:", json.dumps(input_data, indent=2))
        
        # Bước 2: Generate commands
        output = generate_commands(input_data)
        print("Generated commands:", json.dumps(output, indent=2))
        
        # Bước 3: Apply (simulate)
        current_devices = input_data['devices']
        new_devices = apply_commands(output['commands'], current_devices)
        
        # Bước 4: Log
        log_output(input_data, output, new_devices)
        
    except Exception as e:
        print(f"Error in pipeline: {e}")

if __name__ == "__main__":
    main()