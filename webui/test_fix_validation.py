import requests
import json

# 测试修复后的功能
test_cases = [
    {
        "name": "测试修复后的日期参数处理 - 空字符串",
        "data": {
            "file_path": "stock_600525_live",
            "lookback": 100,
            "pred_len": 30,
            "start_date": "",
            "temperature": 1.3
        }
    },
    {
        "name": "测试修复后的日期参数处理 - null字符串",
        "data": {
            "file_path": "stock_600525_live",
            "lookback": 100,
            "pred_len": 30,
            "start_date": "null",
            "temperature": 1.3
        }
    },
    {
        "name": "测试修复后的日期参数处理 - 无start_date参数",
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

print("验证修复后的功能...")
for case in test_cases:
    print(f"\n{case['name']}:")
    try:
        # 发送数据
        data = case['data']
        print(f"发送数据: {json.dumps(data, indent=2, ensure_ascii=False)}")
        response = requests.post(url, json=data)
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            print("✅ 请求成功")
            result = response.json()
            print(f"消息: {result.get('message', 'N/A')}")
        else:
            print(f"❌ 请求失败: {response.text}")
    except Exception as e:
        print(f"❌ 请求异常: {e}")

print("\n验证完成")