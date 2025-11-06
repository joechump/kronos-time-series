import requests
import json

def test_api_endpoint(data, test_name):
    """测试API端点的通用函数"""
    print(f"\n=== {test_name} ===")
    print(f"请求数据: {json.dumps(data, indent=2, ensure_ascii=False)}")
    
    try:
        response = requests.post(
            "http://127.0.0.1:7070/api/predict", 
            json=data, 
            timeout=15
        )
        
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            print("✅ SUCCESS")
            try:
                response_data = response.json()
                print(f"响应数据预览: {json.dumps(response_data, indent=2, ensure_ascii=False)[:200]}...")
            except:
                print(f"响应文本预览: {response.text[:200]}...")
        else:
            print(f"❌ FAILED: {response.text}")
        return response.status_code == 200
        
    except requests.exceptions.ConnectionError as e:
        print("❌ CONNECTION ERROR: 无法连接到服务器")
        return False
    except requests.exceptions.Timeout as e:
        print("❌ TIMEOUT: 请求超时")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

# 测试用例1: start_date为"null"字符串（主要修复）
test1_data = {
    "file_path": "stock_600519_live",
    "lookback": 30,
    "pred_len": 5,
    "start_date": "null"
}
test_api_endpoint(test1_data, "测试用例1: start_date为'null'字符串")

# 测试用例2: start_date为正常日期
test2_data = {
    "file_path": "stock_600519_live",
    "lookback": 30,
    "pred_len": 5,
    "start_date": "2025-01-01"
}
test_api_endpoint(test2_data, "测试用例2: start_date为正常日期")

# 测试用例3: 不提供start_date参数
test3_data = {
    "file_path": "stock_600519_live",
    "lookback": 30,
    "pred_len": 5
}
test_api_endpoint(test3_data, "测试用例3: 不提供start_date参数")

print("\n=== 测试完成 ===")