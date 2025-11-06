import requests
import json

# 测试数据
data = {
    "file_path": "stock_600519_live",
    "lookback": 30,
    "pred_len": 5,
    "start_date": "null"
}

print("=== 测试用例: start_date为null字符串 ===")
print(f"请求URL: http://127.0.0.1:7070/api/predict")
print(f"请求数据: {json.dumps(data, indent=2, ensure_ascii=False)}")

try:
    print("正在发送请求...")
    response = requests.post(
        "http://127.0.0.1:7070/api/predict", 
        json=data, 
        timeout=30
    )
    
    print(f"响应状态码: {response.status_code}")
    print(f"响应头: {dict(response.headers)}")
    
    if response.status_code == 200:
        print("✅ SUCCESS: start_date为'null'的处理已正确修复!")
        try:
            response_data = response.json()
            print(f"响应数据预览: {json.dumps(response_data, indent=2, ensure_ascii=False)[:300]}...")
        except:
            print(f"响应文本预览: {response.text[:300]}...")
    else:
        print(f"❌ FAILED: 状态码 {response.status_code}")
        print(f"错误信息: {response.text}")
        
except requests.exceptions.ConnectionError as e:
    print(f"❌ CONNECTION ERROR: 无法连接到服务器")
    print(f"错误详情: {e}")
    
except requests.exceptions.Timeout as e:
    print(f"❌ TIMEOUT ERROR: 请求超时")
    print(f"错误详情: {e}")
    
except Exception as e:
    print(f"❌ UNEXPECTED ERROR: {e}")
    import traceback
    traceback.print_exc()