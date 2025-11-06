import requests
import json

# 测试用例1: start_date为"null"字符串
print("=== 测试用例: start_date为null字符串 ===")
data1 = {
    "file_path": "stock_600519_live",
    "lookback": 30,
    "pred_len": 5,
    "start_date": "null"
}

try:
    response1 = requests.post("http://127.0.0.1:7070/api/predict", json=data1)
    print(f"发送数据: {json.dumps(data1, ensure_ascii=False)}")
    print(f"状态码: {response1.status_code}")
    print(f"响应: {response1.text}")
except Exception as e:
    print(f"请求失败: {e}")

print("\n" + "="*50 + "\n")

# 测试用例2: start_date为None
print("=== 测试用例: start_date为None ===")
data2 = {
    "file_path": "stock_600519_live",
    "lookback": 30,
    "pred_len": 5,
    "start_date": None
}

try:
    response2 = requests.post("http://127.0.0.1:7070/api/predict", json=data2)
    print(f"发送数据: {json.dumps(data2, ensure_ascii=False)}")
    print(f"状态码: {response2.status_code}")
    print(f"响应: {response2.text}")
except Exception as e:
    print(f"请求失败: {e}")

print("\n" + "="*50 + "\n")

# 测试用例3: 不包含start_date字段
print("=== 测试用例: 不包含start_date字段 ===")
data3 = {
    "file_path": "stock_600519_live",
    "lookback": 30,
    "pred_len": 5
}

try:
    response3 = requests.post("http://127.0.0.1:7070/api/predict", json=data3)
    print(f"发送数据: {json.dumps(data3, ensure_ascii=False)}")
    print(f"状态码: {response3.status_code}")
    print(f"响应: {response3.text}")
except Exception as e:
    print(f"请求失败: {e}")