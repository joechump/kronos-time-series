import requests
import sys

# 测试数据
data = {
    "file_path": "stock_600519_live",
    "lookback": 30,
    "pred_len": 5,
    "start_date": "null"
}

print("正在测试 start_date='null' 的处理...")

try:
    response = requests.post(
        "http://127.0.0.1:7070/api/predict", 
        json=data, 
        timeout=10
    )
    
    if response.status_code == 200:
        print("✅ SUCCESS: start_date为'null'的处理已正确修复!")
        sys.exit(0)  # 成功退出
    else:
        print(f"❌ FAILED: 状态码 {response.status_code}")
        sys.exit(1)  # 失败退出
        
except Exception as e:
    print(f"❌ ERROR: {e}")
    sys.exit(1)  # 失败退出