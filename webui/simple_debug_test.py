import requests
import json

# 简单测试脚本，用于调试后端API对start_date参数的处理

def test_api():
    url = "http://127.0.0.1:7070/api/predict"
    
    # 测试用例: start_date为null字符串
    print("=== 测试用例: start_date为null字符串 ===")
    data = {
        "file_path": "stock_600519_live",
        "lookback": 30,
        "pred_len": 5,
        "start_date": "null"
    }
    print(f"发送数据: {json.dumps(data, ensure_ascii=False)}")
    try:
        response = requests.post(url, json=data)
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.text}")
    except Exception as e:
        print(f"请求失败: {e}")

if __name__ == "__main__":
    test_api()