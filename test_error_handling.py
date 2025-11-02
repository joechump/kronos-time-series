import requests
import json

# 测试错误处理
print("测试错误处理...")

# 1. 测试无效的股票代码
print("\n1. 测试无效的股票代码:")
try:
    response = requests.post('http://localhost:7070/api/akshare/search-stock', 
                           json={'keyword': 'invalid_code'})
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
except Exception as e:
    print(f"错误: {e}")

# 2. 测试空关键词
print("\n2. 测试空关键词:")
try:
    response = requests.post('http://localhost:7070/api/akshare/search-stock', 
                           json={'keyword': ''})
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
except Exception as e:
    print(f"错误: {e}")

# 3. 测试网络错误（关闭服务器后测试）
print("\n3. 测试网络错误:")
try:
    response = requests.post('http://localhost:7070/api/akshare/search-stock', 
                           json={'keyword': '600135'})
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
except Exception as e:
    print(f"网络错误: {e}")

print("\n测试完成")