import subprocess
import json

# 使用PowerShell的Invoke-WebRequest测试API
command = [
    "powershell", "-Command",
    "Invoke-WebRequest -Uri http://127.0.0.1:7070/api/predict -Method POST -ContentType 'application/json' -Body '{\"file_path\": \"stock_600519_live\", \"lookback\": 30, \"pred_len\": 5, \"start_date\": \"null\"}'"
]

try:
    print("Sending request with start_date='null' using PowerShell...")
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    print(f"Return code: {result.returncode}")
    print(f"Stdout: {result.stdout}")
    if result.stderr:
        print(f"Stderr: {result.stderr}")
except Exception as e:
    print(f"Error: {e}")