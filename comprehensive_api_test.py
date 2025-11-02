import requests
import json
import time

def test_api_endpoint(url, method='GET', payload=None, description=""):
    """测试API端点的通用函数"""
    print(f"\n=== {description} ===")
    print(f"URL: {url}")
    print(f"方法: {method}")
    
    try:
        if method == 'GET':
            response = requests.get(url, timeout=30)
        elif method == 'POST' and payload:
            headers = {"Content-Type": "application/json"}
            response = requests.post(url, data=json.dumps(payload), headers=headers, timeout=30)
        else:
            print("不支持的请求方法或缺少负载数据")
            return None
            
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            try:
                response_data = response.json()
                print("✅ 请求成功")
                return response_data
            except json.JSONDecodeError:
                print("✅ 请求成功 (非JSON响应)")
                print("响应内容:", response.text[:200] + "..." if len(response.text) > 200 else response.text)
                return response.text
        else:
            print(f"❌ 请求失败")
            print("响应内容:", response.text[:200] + "..." if len(response.text) > 200 else response.text)
            return None
            
    except requests.exceptions.ConnectionError as e:
        print(f"❌ 连接错误: {e}")
        return None
    except Exception as e:
        print(f"❌ 请求异常: {e}")
        return None

def main():
    base_url = "http://localhost:8080"
    
    print("开始综合API测试...")
    
    # 1. 测试根路径
    test_api_endpoint(f"{base_url}/", "GET", description="测试根路径")
    
    # 2. 测试模型状态
    model_status = test_api_endpoint(f"{base_url}/api/model-status", "GET", description="检查模型状态")
    if model_status:
        print(f"模型可用: {model_status.get('available', 'N/A')}")
        print(f"模型已加载: {model_status.get('loaded', 'N/A')}")
        if model_status.get('current_model'):
            print(f"当前模型: {model_status['current_model'].get('name', 'N/A')}")
    
    # 3. 测试可用模型
    available_models = test_api_endpoint(f"{base_url}/api/available-models", "GET", description="检查可用模型")
    if available_models and available_models.get('models'):
        print("可用模型:")
        for model_key, model_info in available_models['models'].items():
            print(f"  - {model_info.get('name', model_key)}: {model_info.get('description', 'N/A')}")
    
    # 4. 测试股票数据获取
    stock_payload = {
        "symbol": "000001",
        "period": "100d",
        "start_date": "2025-01-01",
        "end_date": "2025-12-31"
    }
    stock_data = test_api_endpoint(f"{base_url}/api/akshare/get-stock-data", "POST", stock_payload, "获取股票数据")
    if stock_data and stock_data.get('data_info'):
        print(f"数据点数量: {len(stock_data.get('data_info', []))}")
        if stock_data.get('data_info'):
            print("数据字段:", list(stock_data['data_info'][0].keys()) if stock_data['data_info'] else "无数据")
    
    # 5. 测试预测API
    predict_payload = {
        "file_path": "stock_000001_live",
        "lookback": 100,
        "pred_len": 30,
        "temperature": 1.0,
        "top_p": 0.9,
        "sample_count": 1
    }
    prediction = test_api_endpoint(f"{base_url}/api/predict", "POST", predict_payload, "股票价格预测")
    if prediction and prediction.get('success'):
        print(f"预测点数: {len(prediction.get('prediction_results', []))}")
        print(f"预测类型: {prediction.get('prediction_type', 'N/A')}")
    
    # 6. 测试交易日历
    calendar_data = test_api_endpoint(f"{base_url}/api/akshare/trading-calendar", "GET", description="获取交易日历")
    
    # 7. 测试是否为交易日
    is_trading_day = test_api_endpoint(f"{base_url}/api/akshare/is-trading-day", "GET", description="检查是否为交易日")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    main()