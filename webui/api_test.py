import requests
import json

# 测试API
def test_api():
    url = "http://127.0.0.1:7070/api/predict"
    
    # 测试数据
    data = {
        "file_path": "stock_600519_live",
        "lookback": 30,
        "pred_len": 5
    }
    
    print("Testing API with corrected parameters...")
    try:
        response = requests.post(url, json=data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_api()