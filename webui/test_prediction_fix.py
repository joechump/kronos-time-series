import requests
import json

# 测试参数
url = "http://localhost:7070/api/predict"
headers = {"Content-Type": "application/json"}

# 测试1: 不带start_date参数(应该正常工作)
test_data_1 = {
    "file_path": "stock_600519_live",
    "lookback": 100,
    "pred_len": 30,
    "temperature": 1.3,
    "top_p": 0.98,
    "sample_count": 2
}

# 测试2: 带空字符串start_date参数(应该正常工作)
test_data_2 = {
    "file_path": "stock_600519_live",
    "lookback": 100,
    "pred_len": 30,
    "temperature": 1.3,
    "top_p": 0.98,
    "sample_count": 2,
    "start_date": ""
}

# 测试3: 带有效日期的start_date参数(应该正常工作)
test_data_3 = {
    "file_path": "stock_600519_live",
    "lookback": 100,
    "pred_len": 30,
    "temperature": 1.3,
    "top_p": 0.98,
    "sample_count": 2,
    "start_date": "2023-01-01"
}

# 执行测试
def run_test(test_data, test_name):
    print(f"\n=== {test_name} ===")
    try:
        response = requests.post(url, headers=headers, data=json.dumps(test_data))
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            print("测试成功: 预测请求已成功处理")
            result = response.json()
            print(f"返回消息: {result.get('message', 'N/A')}")
        else:
            print(f"测试失败: {response.text}")
    except Exception as e:
        print(f"请求异常: {e}")

# 运行所有测试
if __name__ == "__main__":
    print("开始测试预测功能修复...")
    run_test(test_data_1, "测试1: 不带start_date参数")
    run_test(test_data_2, "测试2: 带空字符串start_date参数")
    run_test(test_data_3, "测试3: 带有效日期的start_date参数")
    print("\n所有测试完成。")