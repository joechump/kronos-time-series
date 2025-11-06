import requests
import json

# 测试参数
url = "http://localhost:7070/api/predict"
headers = {"Content-Type": "application/json"}

# 模拟前端可能发送的各种日期参数情况
test_cases = [
    {
        "name": "正常情况 - 不带start_date参数",
        "data": {
            "file_path": "stock_600519_live",
            "lookback": 100,
            "pred_len": 30,
            "temperature": 1.3,
            "top_p": 0.98,
            "sample_count": 2
        }
    },
    {
        "name": "正常情况 - start_date为null",
        "data": {
            "file_path": "stock_600519_live",
            "lookback": 100,
            "pred_len": 30,
            "temperature": 1.3,
            "top_p": 0.98,
            "sample_count": 2,
            "start_date": None
        }
    },
    {
        "name": "修复验证 - start_date为'null'字符串",
        "data": {
            "file_path": "stock_600519_live",
            "lookback": 100,
            "pred_len": 30,
            "temperature": 1.3,
            "top_p": 0.98,
            "sample_count": 2,
            "start_date": "null"
        }
    },
    {
        "name": "修复验证 - start_date为'undefined'字符串",
        "data": {
            "file_path": "stock_600519_live",
            "lookback": 100,
            "pred_len": 30,
            "temperature": 1.3,
            "top_p": 0.98,
            "sample_count": 2,
            "start_date": "undefined"
        }
    },
    {
        "name": "修复验证 - start_date为空字符串",
        "data": {
            "file_path": "stock_600519_live",
            "lookback": 100,
            "pred_len": 30,
            "temperature": 1.3,
            "top_p": 0.98,
            "sample_count": 2,
            "start_date": ""
        }
    },
    {
        "name": "正常情况 - start_date为有效日期",
        "data": {
            "file_path": "stock_600519_live",
            "lookback": 100,
            "pred_len": 30,
            "temperature": 1.3,
            "top_p": 0.98,
            "sample_count": 2,
            "start_date": "2023-01-01"
        }
    }
]

# 执行测试
def run_test(test_case):
    print(f"\n=== {test_case['name']} ===")
    print(f"请求参数: {json.dumps(test_case['data'], indent=2, ensure_ascii=False)}")
    try:
        response = requests.post(url, headers=headers, json=test_case['data'])
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            print("✅ 测试成功: 预测请求已成功处理")
            result = response.json()
            print(f"返回消息: {result.get('message', 'N/A')}")
        else:
            print(f"❌ 测试失败: {response.text}")
            try:
                error_data = response.json()
                print(f"错误详情: {error_data}")
            except:
                print("无法解析错误响应")
    except Exception as e:
        print(f"❌ 请求异常: {e}")

# 运行所有测试
if __name__ == "__main__":
    print("开始测试前端修复后的参数处理逻辑...")
    for test_case in test_cases:
        run_test(test_case)
    print("\n所有测试完成。")