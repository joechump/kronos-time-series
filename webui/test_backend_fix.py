import requests
import json

# 测试后端修复
def test_backend_fix():
    url = "http://127.0.0.1:7070/api/predict"
    
    # 测试数据
    test_cases = [
        {
            "name": "Valid date",
            "data": {
                "model_key": "kronos-small",
                "device": "cpu",
                "stock_code": "600519",
                "lookback": 30,
                "pred_len": 5,
                "trading_mode": False,
                "start_date": "2025-01-01"
            }
        },
        {
            "name": "Null string start_date",
            "data": {
                "model_key": "kronos-small",
                "device": "cpu",
                "stock_code": "600519",
                "lookback": 30,
                "pred_len": 5,
                "trading_mode": False,
                "start_date": "null"
            }
        },
        {
            "name": "Undefined string start_date",
            "data": {
                "model_key": "kronos-small",
                "device": "cpu",
                "stock_code": "600519",
                "lookback": 30,
                "pred_len": 5,
                "trading_mode": False,
                "start_date": "undefined"
            }
        },
        {
            "name": "Empty string start_date",
            "data": {
                "model_key": "kronos-small",
                "device": "cpu",
                "stock_code": "600519",
                "lookback": 30,
                "pred_len": 5,
                "trading_mode": False,
                "start_date": ""
            }
        },
        {
            "name": "No start_date parameter",
            "data": {
                "model_key": "kronos-small",
                "device": "cpu",
                "stock_code": "600519",
                "lookback": 30,
                "pred_len": 5,
                "trading_mode": False
            }
        }
    ]
    
    print("Testing backend fixes...")
    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        try:
            response = requests.post(url, json=test_case['data'])
            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                print("Result: SUCCESS")
            else:
                print(f"Result: FAILED - {response.text}")
        except Exception as e:
            print(f"Result: ERROR - {str(e)}")

if __name__ == "__main__":
    test_backend_fix()