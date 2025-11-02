import requests
import json
import time

# API endpoint
url = "http://localhost:8080/api/predict"

# Request payload for real-time stock prediction
payload = {
    "file_path": "stock_000001_live",  # 实时股票数据请求格式
    "lookback": 400,                   # 回看数据点数
    "pred_len": 120,                   # 预测数据点数
    "temperature": 1.0,                # 预测温度参数
    "top_p": 0.9,                      # 核心采样参数
    "sample_count": 1                  # 样本数量
}

# Headers
headers = {
    "Content-Type": "application/json"
}

print("发送预测请求到:", url)
print("请求参数:", json.dumps(payload, indent=2, ensure_ascii=False))

try:
    # Send POST request
    response = requests.post(url, data=json.dumps(payload), headers=headers, timeout=60)
    
    # Parse response
    print("\n响应状态码:", response.status_code)
    
    if response.status_code == 200:
        response_data = response.json()
        print("\n完整响应键:", list(response_data.keys()))
        
        if response_data.get('success', False):
            print("\n✅ 预测成功!")
            print("预测类型:", response_data.get('prediction_type', 'N/A'))
            print("消息:", response_data.get('message', 'N/A'))
            
            # 检查预测结果
            prediction_results = response_data.get('prediction_results', [])
            print(f"预测结果数量: {len(prediction_results)}")
            
            if prediction_results:
                print("\n前5个预测点:")
                for i, point in enumerate(prediction_results[:5]):
                    print(f"  {i+1}. 时间: {point.get('timestamp', 'N/A')}")
                    print(f"     开盘价: {point.get('open', 'N/A')}")
                    print(f"     最高价: {point.get('high', 'N/A')}")
                    print(f"     最低价: {point.get('low', 'N/A')}")
                    print(f"     收盘价: {point.get('close', 'N/A')}")
        else:
            print("\n❌ 预测失败!")
            print("错误信息:", response_data.get('error', 'N/A'))
    else:
        print("请求失败，状态码:", response.status_code)
        print("响应内容:", response.text)
        
except requests.exceptions.ConnectionError as e:
    print("连接错误:", e)
    print("请确保服务器正在运行且端口正确")
except Exception as e:
    print("请求异常:", e)
    print("异常类型:", type(e).__name__)