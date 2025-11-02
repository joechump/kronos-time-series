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
        print("\n完整响应:", json.dumps(response_data, indent=2, ensure_ascii=False))
        
        # 详细检查各个字段
        print("\n=== 详细状态信息 ===")
        print("健康状态:", response_data.get('healthy', 'N/A'))
        print("是否可用:", response_data.get('available', 'N/A'))
        print("是否加载:", response_data.get('loaded', 'N/A'))
        print("直接模型加载:", response_data.get('direct_model_loaded', 'N/A'))
        print("自动加载启用:", response_data.get('auto_load_enabled', 'N/A'))
        print("消息:", response_data.get('message', 'N/A'))
        print("时间戳:", response_data.get('timestamp', 'N/A'))
        
        # 检查当前模型信息
        current_model = response_data.get('current_model', {})
        if current_model:
            print("\n=== 当前模型信息 ===")
            print("模型名称:", current_model.get('name', 'N/A'))
            print("模型ID:", current_model.get('model_id', 'N/A'))
            print("参数:", current_model.get('params', 'N/A'))
            print("上下文长度:", current_model.get('context_length', 'N/A'))
            print("描述:", current_model.get('description', 'N/A'))
        else:
            print("\n当前没有加载模型")
            
        # 检查直接模型信息
        direct_model_info = response_data.get('direct_model_info', {})
        if direct_model_info:
            print("\n=== 直接模型信息 ===")
            print("模型名称:", direct_model_info.get('name', 'N/A'))
            print("加载时间:", direct_model_info.get('loaded_at', 'N/A'))
            print("设备:", direct_model_info.get('device', 'N/A'))
        else:
            print("\n没有直接模型信息")
            
        # 检查系统信息
        system_info = response_data.get('system_info', {})
        if system_info:
            print("\n=== 系统信息 ===")
            print("CPU核心数:", system_info.get('cpu_cores', 'N/A'))
            print("内存总量:", system_info.get('total_memory', 'N/A'))
            print("可用内存:", system_info.get('available_memory', 'N/A'))
            print("平台:", system_info.get('platform', 'N/A'))
            
    else:
        print("请求失败，状态码:", response.status_code)
        print("响应内容:", response.text)
        
except requests.exceptions.ConnectionError as e:
    print("连接错误:", e)
    print("请确保服务器正在运行且端口正确")
except Exception as e:
    print("请求异常:", e)
    print("异常类型:", type(e).__name__)
    import traceback
    traceback.print_exc()