import requests
import json

# API endpoint for available models
url = "http://localhost:8080/api/available-models"

print("检查可用模型:", url)

try:
    # Send GET request
    response = requests.get(url, timeout=30)
    
    # Parse response
    print("\n响应状态码:", response.status_code)
    
    if response.status_code == 200:
        response_data = response.json()
        print("\n完整响应:", json.dumps(response_data, indent=2, ensure_ascii=False))
        
        # 检查可用模型
        available_models = response_data.get('available_models', {})
        if available_models:
            print("\n=== 可用模型列表 ===")
            for model_key, model_info in available_models.items():
                print(f"\n模型: {model_key}")
                print(f"  名称: {model_info.get('name', 'N/A')}")
                print(f"  描述: {model_info.get('description', 'N/A')}")
                print(f"  参数: {model_info.get('params', 'N/A')}")
                print(f"  上下文长度: {model_info.get('context_length', 'N/A')}")
                print(f"  GPU要求: {model_info.get('gpu_required', 'N/A')}")
                print(f"  最小内存: {model_info.get('min_memory_gb', 'N/A')} GB")
                print(f"  模型ID: {model_info.get('model_id', 'N/A')}")
                print(f"  分词器ID: {model_info.get('tokenizer_id', 'N/A')}")
        else:
            print("\n没有可用模型信息")
            
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