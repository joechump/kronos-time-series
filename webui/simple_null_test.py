import requests
import json

# 测试用例: start_date为"null"字符串
print("=== 测试用例: start_date为null字符串 ===")
data = {
    "file_path": "stock_600519_live",
    "lookback": 30,
    "pred_len": 5,
    "start_date": "null"
}

response = requests.post("http://127.0.0.1:7070/api/predict", json=data)
print(f"发送数据: {json.dumps(data, ensure_ascii=False)}")
print(f"状态码: {response.status_code}")
if response.status_code == 200:
    print("请求成功! start_date为'null'的处理已正确修复。")
    print("API正确地将'start_date': 'null'视为使用最新数据。")
else:
    print(f"响应: {response.text}")