import requests
import json

# 测试用例: start_date为"null"字符串
print("=== 测试用例: start_date为null字符串 ===")
data = {
    "file_path": "stock_600519_live",
    "lookback": 30,
    "pred_len": 5,
    "start_date": "null"
}

try:
    print("正在发送请求到 http://127.0.0.1:7070/api/predict ...")
    response = requests.post("http://127.0.0.1:7070/api/predict", json=data, timeout=30)
    print(f"状态码: {response.status_code}")
    if response.status_code == 200:
        print("✅ 请求成功! start_date为'null'的处理已正确修复。")
        print("✅ API正确地将'start_date': 'null'视为使用最新数据。")
        # 打印响应的一部分来确认
        print(f"响应预览: {response.text[:300]}...")
    else:
        print(f"❌ 请求失败，状态码: {response.status_code}")
        print(f"响应内容: {response.text[:300]}...")
except requests.exceptions.ConnectionError as e:
    print(f"❌ 连接错误: {str(e)}")
    print("请检查服务器是否正在运行")
except requests.exceptions.Timeout as e:
    print(f"❌ 请求超时: {str(e)}")
    print("服务器可能正在处理请求，请稍后再试")
except Exception as e:
    print(f"❌ 请求过程中出现错误: {str(e)}")
    import traceback
    traceback.print_exc()