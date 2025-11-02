import requests
import json

# API endpoint for model status
url = "http://localhost:8080/api/model-status"

print("检查模型状态:", url)

try:
    # Send GET request
    response = requests.get(url, timeout=30)
    
    # Parse response
    print("\n响应状态码:", response.status_code)
    
    if response.status_code == 200:
        response_data = response.json()
        print("\n完整响应键:", list(response_data.keys()))
        
        if response_data.get('model_available', False):
            print("\n✅ 模型可用!")
            print("模型名称:", response_data.get('model_name', 'N/A'))
            print("模型参数:", response_data.get('model_params', 'N/A'))
            print("设备:", response_data.get('device', 'N/A'))
            print("上下文长度:", response_data.get('context_length', 'N/A'))
            print("加载时间:", response_data.get('loaded_at', 'N/A'))
        else:
            print("\n❌ 模型不可用!")
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