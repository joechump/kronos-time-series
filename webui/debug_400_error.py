import requests
import json
import time

def test_predict_api():
    """测试预测API，找出400错误的具体原因"""
    
    # 等待服务器启动
    print("等待服务器启动...")
    time.sleep(3)
    
    # 测试参数 - 模拟前端请求
    url = "http://localhost:8080/api/predict"
    
    # 使用前端实际发送的参数
    params = {
        "file_path": "stock_600519_live",
        "lookback": 400,
        "pred_len": 120,
        "start_date": "",
        "end_date": "",
        "model_name": "kronos-small"
    }
    
    print(f"测试参数: {params}")
    
    try:
        # 发送请求
        response = requests.post(url, json=params, timeout=30)
        
        print(f"状态码: {response.status_code}")
        print(f"响应头: {response.headers}")
        
        if response.status_code == 200:
            print("✅ 请求成功!")
            result = response.json()
            print(f"响应内容: {json.dumps(result, indent=2, ensure_ascii=False)}")
        else:
            print(f"❌ 请求失败，状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            
            # 尝试获取更详细的错误信息
            if response.status_code == 400:
                print("\n=== 400错误分析 ===")
                print("可能的原因:")
                print("1. 模型未加载")
                print("2. 数据长度不足")
                print("3. 日期格式错误")
                print("4. 参数验证失败")
                print("5. 数据提供者错误")
                
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求异常: {e}")
    except Exception as e:
        print(f"❌ 其他异常: {e}")

def test_model_status():
    """测试模型状态API"""
    url = "http://localhost:8080/api/model-status"
    
    try:
        response = requests.get(url, timeout=10)
        print(f"\n模型状态测试:")
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            print(f"模型状态: {response.json()}")
        else:
            print(f"错误: {response.text}")
    except Exception as e:
        print(f"模型状态测试失败: {e}")

def test_data_provider():
    """测试数据提供者"""
    url = "http://localhost:8080/api/stock-info"
    params = {"symbol": "600519"}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        print(f"\n数据提供者测试:")
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            print(f"股票信息: {response.json()}")
        else:
            print(f"错误: {response.text}")
    except Exception as e:
        print(f"数据提供者测试失败: {e}")

if __name__ == "__main__":
    print("开始诊断400错误...")
    test_model_status()
    test_data_provider()
    test_predict_api()