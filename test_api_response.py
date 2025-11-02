import requests
import json

# 测试股票搜索API
def test_search_stock_api():
    url = "http://localhost:8080/api/akshare/search-stock"
    
    # 测试数据
    test_cases = [
        {"keyword": "平安银行"},
        {"keyword": "000001"},
        {"keyword": "600519"},
        {"keyword": "贵州茅台"}
    ]
    
    for case in test_cases:
        print(f"\n测试搜索: {case['keyword']}")
        try:
            response = requests.post(url, json=case)
            print(f"状态码: {response.status_code}")
            print(f"响应头: {dict(response.headers)}")
            
            # 尝试解析JSON响应
            try:
                data = response.json()
                print(f"响应数据: {json.dumps(data, ensure_ascii=False, indent=2)}")
            except:
                print(f"响应内容: {response.text[:500]}...")
                
        except Exception as e:
            print(f"请求失败: {e}")

# 测试股票数据API
def test_get_stock_data_api():
    url = "http://localhost:8080/api/akshare/get-stock-data"
    
    # 测试数据
    test_cases = [
        {"symbol": "sz000001", "period": "100d"},
        {"symbol": "sh600519", "period": "100d"}
    ]
    
    for case in test_cases:
        print(f"\n测试获取股票数据: {case}")
        try:
            response = requests.post(url, json=case)
            print(f"状态码: {response.status_code}")
            print(f"响应头: {dict(response.headers)}")
            
            # 尝试解析JSON响应
            try:
                data = response.json()
                print(f"响应数据: {json.dumps(data, ensure_ascii=False, indent=2)}")
            except:
                print(f"响应内容: {response.text[:500]}...")
                
        except Exception as e:
            print(f"请求失败: {e}")

if __name__ == "__main__":
    print("测试API端点响应格式")
    test_search_stock_api()
    test_get_stock_data_api()