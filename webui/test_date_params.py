import requests
import json

# 测试参数
url = "http://localhost:7070/api/predict"
headers = {"Content-Type": "application/json"}

# 测试用例
test_cases = [
    {
        "name": "有效日期格式",
        "data": {
            "file_path": "stock_600519_live",
            "start_date": "2023-01-01"
        }
    },
    {
        "name": "无效日期格式",
        "data": {
            "file_path": "stock_600519_live",
            "start_date": "invalid-date"
        }
    },
    {
        "name": "null字符串",
        "data": {
            "file_path": "stock_600519_live",
            "start_date": "null"
        }
    },
    {
        "name": "undefined字符串",
        "data": {
            "file_path": "stock_600519_live",
            "start_date": "undefined"
        }
    },
    {
        "name": "空字符串",
        "data": {
            "file_path": "stock_600519_live",
            "start_date": ""
        }
    },
    {
        "name": "不带start_date参数",
        "data": {
            "file_path": "stock_600519_live"
        }
    }
]

print("开始测试各种日期参数处理情况...\n")

for i, test_case in enumerate(test_cases, 1):
    print(f"{i}. {test_case['name']}:")
    print(f"   请求参数: {test_case['data']}")
    
    try:
        response = requests.post(url, headers=headers, json=test_case['data'])
        print(f"   状态码: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ 成功")
            result = response.json()
            print(f"   消息: {result.get('message', 'N/A')}")
        else:
            print("   ❌ 失败")
            try:
                error_data = response.json()
                print(f"   错误: {error_data.get('error', 'N/A')}")
            except:
                print(f"   错误: {response.text}")
    except Exception as e:
        print(f"   ❌ 异常: {e}")
    
    print()

print("测试完成。")