import requests
import json

def test_stock_data_api():
    """测试股票数据API"""
    url = "http://localhost:8080/api/akshare/get-stock-data"
    payload = {
        "symbol": "sz000001",  # 使用带市场前缀的股票代码
        "period": "100d",
        "start_date": "2025-01-01",
        "end_date": "2025-12-31"
    }
    
    headers = {"Content-Type": "application/json"}
    
    print("测试股票数据API")
    print("URL:", url)
    print("请求参数:", json.dumps(payload, indent=2, ensure_ascii=False))
    
    try:
        response = requests.post(url, data=json.dumps(payload), headers=headers, timeout=30)
        print("\n响应状态码:", response.status_code)
        
        if response.status_code == 200:
            response_data = response.json()
            print("✅ 请求成功")
            print("响应字段:", list(response_data.keys()))
            
            if 'data_info' in response_data:
                data_info = response_data['data_info']
                print(f"数据点数量: {len(data_info)}")
                if data_info:
                    print("第一个数据点字段:", list(data_info[0].keys()))
                    print("前3个数据点:")
                    for i, point in enumerate(data_info[:3]):
                        print(f"  {i+1}. 日期: {point.get('date', 'N/A')}")
                        print(f"     开盘价: {point.get('open', 'N/A')}")
                        print(f"     最高价: {point.get('high', 'N/A')}")
                        print(f"     最低价: {point.get('low', 'N/A')}")
                        print(f"     收盘价: {point.get('close', 'N/A')}")
                        print(f"     成交量: {point.get('volume', 'N/A')}")
            else:
                print("响应中没有data_info字段")
                
            if 'price_range' in response_data:
                print("价格范围:", response_data['price_range'])
            else:
                print("响应中没有price_range字段")
        else:
            print("❌ 请求失败")
            print("响应内容:", response.text)
            
    except requests.exceptions.ConnectionError as e:
        print("连接错误:", e)
    except Exception as e:
        print("请求异常:", e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_stock_data_api()