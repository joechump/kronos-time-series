import requests
import json

# 测试不同的start_date参数值
test_cases = [
    {
        "name": "测试有效的start_date参数",
        "data": {
            "file_path": "stock_600525_live",
            "lookback": 100,
            "pred_len": 30,
            "start_date": "2025-11-04",
            "temperature": 1.3
        }
    },
    {
        "name": "测试空字符串start_date参数",
        "data": {
            "file_path": "stock_600525_live",
            "lookback": 100,
            "pred_len": 30,
            "start_date": "",
            "temperature": 1.3
        }
    },
    {
        "name": "测试None值start_date参数",
        "data": {
            "file_path": "stock_600525_live",
            "lookback": 100,
            "pred_len": 30,
            "temperature": 1.3
        }
    }
]

# 发送测试请求
url = "http://localhost:7070/api/predict"

for case in test_cases:
    print(f"\n{case['name']}:")
    try:
        # 如果是测试None值的情况，从数据中移除start_date键
        if case['name'] == "测试None值start_date参数":
            data = {k: v for k, v in case['data'].items() if k != 'start_date'}
        else:
            data = case['data']
            
        response = requests.post(url, json=data)
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            print("请求成功")
            result = response.json()
            print(f"消息: {result.get('message', 'N/A')}")
        else:
            print(f"请求失败: {response.text}")
    except Exception as e:
        print(f"请求异常: {e}")