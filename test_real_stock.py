import requests
import json

# 测试真实的股票代码
url = "http://localhost:7070/api/akshare/search-stock"
payload = {
    "keyword": "600135"  # 乐凯胶片
}

try:
    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
    
    if response.status_code == 200:
        response_data = response.json()
        print("Response Body:")
        print(json.dumps(response_data, indent=2, ensure_ascii=False))
        
        # 检查返回的数据
        if response_data.get("success") and response_data.get("results"):
            stock_info = response_data["results"][0]
            print(f"\n股票信息:")
            print(f"  名称: {stock_info.get('name', 'N/A')}")
            print(f"  代码: {stock_info.get('symbol', 'N/A')}")
            print(f"  最新价: {stock_info.get('latest_price', 'N/A')}")
            print(f"  涨跌幅: {stock_info.get('change_rate', 'N/A')}")
            print(f"  涨跌额: {stock_info.get('change_amount', 'N/A')}")
            print(f"  成交量: {stock_info.get('volume', 'N/A')}")
            print(f"  成交额: {stock_info.get('amount', 'N/A')}")
        else:
            print("未找到股票信息或请求失败")
    else:
        print(f"请求失败，状态码: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"请求出错: {e}")
    import traceback
    traceback.print_exc()