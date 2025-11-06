import requests
import json

# 测试参数
url = "http://localhost:7070/api/predict"
headers = {"Content-Type": "application/json"}

# 测试1: 带null字符串的start_date参数(模拟前端可能的问题)
test_data_1 = {
    "file_path": "stock_600519_live",
    "lookback": 100,
    "pred_len": 30,
    "temperature": 1.3,
    "top_p": 0.98,
    "sample_count": 2,
    "start_date": "null"  # 这可能是前端传递的问题参数
}

# 测试2: 带undefined字符串的start_date参数
test_data_2 = {
    "file_path": "stock_600519_live",
    "lookback": 100,
    "pred_len": 30,
    "temperature": 1.3,
    "top_p": 0.98,
    "sample_count": 2,
    "start_date": "undefined"  # 这也可能是前端传递的问题参数
}

# 测试3: 正常的空值参数
test_data_3 = {
    "file_path": "stock_600519_live",
    "lookback": 100,
    "pred_len": 30,
    "temperature": 1.3,
    "top_p": 0.98,
    "sample_count": 2,
    "start_date": None  # 正常的空值
}

# 测试4: 完全不带start_date参数
test_data_4 = {
    "file_path": "stock_600519_live",
    "lookback": 100,
    "pred_len": 30,
    "temperature": 1.3,
    "top_p": 0.98,
    "sample_count": 2
    # 不包含start_date参数
}

# 执行测试
def run_test(test_data, test_name):
    print(f"\n=== {test_name} ===")
    print(f"请求参数: {json.dumps(test_data, indent=2, ensure_ascii=False)}")
    try:
        response = requests.post(url, headers=headers, json=test_data)
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
    print("开始测试预测API参数处理...")
    run_test(test_data_1, "测试1: 带'null'字符串的start_date参数")
    run_test(test_data_2, "测试2: 带'undefined'字符串的start_date参数")
    run_test(test_data_3, "测试3: 带None值的start_date参数")
    run_test(test_data_4, "测试4: 不带start_date参数")
    print("\n所有测试完成。")