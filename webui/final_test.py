import requests
import json

# 测试参数
url = "http://localhost:7070/api/predict"
headers = {"Content-Type": "application/json"}

def test_case(name, data):
    print(f"测试: {name}")
    print(f"请求参数: {data}")
    
    try:
        response = requests.post(url, headers=headers, json=data)
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            print("[成功]")
            result = response.json()
            print(f"消息: {result.get('message', 'N/A')}")
        else:
            print("[失败]")
            try:
                error_data = response.json()
                print(f"错误: {error_data.get('error', 'N/A')}")
            except:
                print(f"错误: {response.text}")
    except Exception as e:
        print(f"[异常]: {e}")
    
    print("-" * 50)

# 分批测试
print("=== 第一批测试 ===")
# 1. 有效日期格式
test_case("有效日期格式", {
    "file_path": "stock_600519_live",
    "start_date": "2023-01-01"
})

# 2. 无效日期格式
test_case("无效日期格式", {
    "file_path": "stock_600519_live",
    "start_date": "invalid-date"
})

input("按回车键继续第二批测试...")

print("\n=== 第二批测试 ===")
# 3. null字符串
test_case("null字符串", {
    "file_path": "stock_600519_live",
    "start_date": "null"
})

# 4. undefined字符串
test_case("undefined字符串", {
    "file_path": "stock_600519_live",
    "start_date": "undefined"
})

input("按回车键继续第三批测试...")

print("\n=== 第三批测试 ===")
# 5. 空字符串
test_case("空字符串", {
    "file_path": "stock_600519_live",
    "start_date": ""
})

# 6. 不带start_date参数
test_case("不带start_date参数", {
    "file_path": "stock_600519_live"
})

print("所有测试完成。")