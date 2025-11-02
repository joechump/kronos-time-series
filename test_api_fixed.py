import requests
import json

# 测试 /api/akshare/search-stock API端点
print("测试 /api/akshare/search-stock API端点\\n")

# 测试关键词列表
test_keywords = ["平安银行", "000001", "600519", "贵州茅台"]

for keyword in test_keywords:
    print(f"测试搜索关键词: {keyword}")
    try:
        # 使用正确的API端点路径
        response = requests.post(
            "http://localhost:8080/api/akshare/search-stock",
            json={"keyword": keyword},
            timeout=10
        )
        
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"成功响应: {json.dumps(data, ensure_ascii=False, indent=2)}")
        else:
            print(f"错误响应: {response.text}")
    except Exception as e:
        print(f"请求失败: {e}")
    print()

# 测试 /api/akshare/get-stock-data API端点
print("\\n测试 /api/akshare/get-stock-data API端点\\n")

# 测试股票代码列表
test_stocks = ["sz000001", "sh600519"]

for stock_code in test_stocks:
    print(f"测试获取股票数据: {stock_code}")
    try:
        # 使用正确的API端点路径
        response = requests.post(
            "http://localhost:8080/api/akshare/get-stock-data",
            json={"symbol": stock_code},
            timeout=10
        )
        
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"成功响应: {json.dumps(data, ensure_ascii=False, indent=2)}")
        else:
            print(f"错误响应: {response.text}")
    except Exception as e:
        print(f"请求失败: {e}")
    print()