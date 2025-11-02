import requests
import json

url = "http://localhost:8080/api/akshare/get-stock-data"
data = {
    "stock_code": "sz000001",
    "start_date": "2023-11-01",
    "end_date": "2023-11-30"
}

print(f"请求URL: {url}")
print(f"请求数据: {data}")

try:
    response = requests.post(url, json=data, timeout=30)
    print(f"状态码: {response.status_code}")
    print(f"响应头: {response.headers}")
    print(f"响应内容: {response.text}")
    
    if response.status_code == 200:
        # 尝试解析JSON响应
        try:
            json_data = response.json()
            print(f"JSON响应: {json.dumps(json_data, indent=2, ensure_ascii=False)}")
            
            # 检查是否有price_range字段
            if 'price_range' in json_data:
                print(f"price_range字段: {json_data['price_range']}")
            else:
                print("响应中没有找到price_range字段")
        except json.JSONDecodeError:
            print("响应不是有效的JSON格式")
    else:
        print(f"请求失败，状态码: {response.status_code}")
        
except Exception as e:
    print(f"请求失败: {e}")