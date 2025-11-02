import requests
import json

# 测试API接口，使用特殊关键字强制触发模拟数据
url = "http://localhost:7070/api/akshare/search-stock"
payload = {
    "keyword": "600135_test"
}

try:
    response = requests.post(url, json=payload, timeout=30)
    print(f"Status Code: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
    print(f"Response Body: {response.text}")
    
    # 尝试解析JSON
    try:
        json_data = response.json()
        print(f"JSON Response: {json.dumps(json_data, indent=2, ensure_ascii=False)}")
    except Exception as json_error:
        print(f"Failed to parse JSON: {json_error}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()