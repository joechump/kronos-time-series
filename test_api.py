import requests
import json

# 测试API接口
url = "http://localhost:7070/api/akshare/search-stock"
payload = {
    "keyword": "600135"
}

try:
    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
except Exception as e:
    print(f"Error: {e}")