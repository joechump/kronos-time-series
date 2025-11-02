import requests
import time

url = "http://localhost:8080/api/akshare/get-stock-data"
params = {
    "stock_code": "sz000001",
    "start_date": "2023-11-01",
    "end_date": "2023-11-30"
}

print(f"请求URL: {url}")
print(f"请求参数: {params}")

try:
    response = requests.get(url, params=params, timeout=30)
    print(f"状态码: {response.status_code}")
    print(f"响应头: {response.headers}")
    print(f"响应内容: {response.text}")
except Exception as e:
    print(f"请求失败: {e}")