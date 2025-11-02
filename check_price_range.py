import requests
import json

# API endpoint
url = "http://localhost:8080/api/akshare/get-stock-data"

# Request payload - 使用正确的字段名
payload = {
    "symbol": "000001",  # 使用股票代码而不是stock_code
    "start_date": "20230101",
    "end_date": "20231231",
    "period": "daily"
}

# Headers
headers = {
    "Content-Type": "application/json"
}

print("发送请求到:", url)
print("请求参数:", json.dumps(payload, indent=2, ensure_ascii=False))

try:
    # Send POST request
    response = requests.post(url, data=json.dumps(payload), headers=headers, timeout=30)
    
    # Parse response
    print("\n响应状态码:", response.status_code)
    
    if response.status_code == 200:
        response_data = response.json()
        print("\n完整响应键:", list(response_data.keys()))
        
        if 'data_info' in response_data:
            data_info = response_data['data_info']
            print("\nData Info键:", list(data_info.keys()))
            
            # Check if price_range exists
            if 'price_range' in data_info:
                print("\n找到price_range字段:")
                print(json.dumps(data_info['price_range'], indent=2, ensure_ascii=False))
                
                # 详细检查每个价格类型
                price_range = data_info['price_range']
                for price_type in ['open', 'high', 'low', 'close']:
                    if price_type in price_range:
                        print(f"\n{price_type}价格范围:")
                        print(f"  最低价: {price_range[price_type]['min']}")
                        print(f"  最高价: {price_range[price_type]['max']}")
                    else:
                        print(f"\n未找到{price_type}价格范围")
            else:
                print("\nprice_range字段未在data_info中找到")
                print("data_info内容:", json.dumps(data_info, indent=2, ensure_ascii=False))
        else:
            print("\ndata_info未在响应中找到")
            print("完整响应:", json.dumps(response_data, indent=2, ensure_ascii=False))
    else:
        print("请求失败，状态码:", response.status_code)
        print("响应内容:", response.text)
        
except requests.exceptions.ConnectionError as e:
    print("连接错误:", e)
    print("请确保服务器正在运行且端口正确")
except Exception as e:
    print("请求异常:", e)