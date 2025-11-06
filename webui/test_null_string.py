import requests
import json

# 测试数据
test_data = {
    "stock_code": "sh.600000",
    "start_date": "null",  # 故意传入"null"字符串
    "end_date": "2024-12-31",
    "target_date": "2025-01-15",
    "model_type": "kronos-small",
    "prediction_days": 30
}

# 发送POST请求
url = "http://127.0.0.1:7070/api/predict"
headers = {"Content-Type": "application/json"}

try:
    response = requests.post(url, data=json.dumps(test_data), headers=headers)
    print(f"状态码: {response.status_code}")
    print(f"响应内容: {response.text}")
    
    # 解析响应
    if response.status_code == 200:
        result = response.json()
        print(f"预测结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
    else:
        print(f"请求失败: {response.status_code}")
        
except Exception as e:
    print(f"请求出错: {e}")