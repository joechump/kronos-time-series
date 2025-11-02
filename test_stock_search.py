import requests
import json

def test_stock_search_api():
    """测试股票代码搜索API"""
    url = "http://localhost:8080/api/akshare/search-stock"
    payload = {
        "keyword": "平安银行"  # 搜索股票名称
    }
    
    headers = {"Content-Type": "application/json"}
    
    print("测试股票代码搜索API")
    print("URL:", url)
    print("请求参数:", json.dumps(payload, indent=2, ensure_ascii=False))
    
    try:
        response = requests.post(url, data=json.dumps(payload), headers=headers, timeout=30)
        print("\n响应状态码:", response.status_code)
        
        if response.status_code == 200:
            response_data = response.json()
            print("✅ 请求成功")
            print("响应字段:", list(response_data.keys()))
            
            # 检查成功状态
            if response_data.get('success', False):
                results = response_data.get('results', [])
                print(f"找到 {len(results)} 只股票:")
                for i, stock in enumerate(results[:5]):  # 只显示前5只
                    print(f"  {i+1}. 代码: {stock.get('code', 'N/A')}")
                    print(f"     名称: {stock.get('name', 'N/A')}")
                    print(f"     市场: {stock.get('market', 'N/A')}")
            else:
                print("搜索失败:", response_data.get('message', '未知错误'))
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
    test_stock_search_api()