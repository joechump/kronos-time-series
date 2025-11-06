import requests

# 最简化的测试
data = {
    "file_path": "stock_600519_live",
    "lookback": 30,
    "pred_len": 5,
    "start_date": "null"
}

print("Sending request with start_date='null'...")
response = requests.post("http://127.0.0.1:7070/api/predict", json=data, timeout=15)
print(f"Status: {response.status_code}")
if response.status_code == 200:
    print("SUCCESS: Fix is working!")
else:
    print(f"FAILED: {response.text[:100]}")