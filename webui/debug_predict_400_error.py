import requests
import json
import sys

# 测试预测API的400错误
url = "http://localhost:8080/api/predict"
headers = {"Content-Type": "application/json"}

# 模拟前端可能发送的各种参数组合
test_cases = [
    {
        "name": "正常股票代码预测",
        "data": {
            "file_path": "stock_600519_live",
            "lookback": 100,
            "pred_len": 30,
            "temperature": 1.3,
            "top_p": 0.98,
            "sample_count": 2
        }
    },
    {
        "name": "带start_date参数的预测",
        "data": {
            "file_path": "stock_600519_live",
            "lookback": 100,
            "pred_len": 30,
            "temperature": 1.3,
            "top_p": 0.98,
            "sample_count": 2,
            "start_date": "2024-01-01"
        }
    },
    {
        "name": "start_date为null字符串",
        "data": {
            "file_path": "stock_600519_live",
            "lookback": 100,
            "pred_len": 30,
            "temperature": 1.3,
            "top_p": 0.98,
            "sample_count": 2,
            "start_date": "null"
        }
    },
    {
        "name": "start_date为undefined字符串",
        "data": {
            "file_path": "stock_600519_live",
            "lookback": 100,
            "pred_len": 30,
            "temperature": 1.3,
            "top_p": 0.98,
            "sample_count": 2,
            "start_date": "undefined"
        }
    },
    {
        "name": "start_date为None",
        "data": {
            "file_path": "stock_600519_live",
            "lookback": 100,
            "pred_len": 30,
            "temperature": 1.3,
            "top_p": 0.98,
            "sample_count": 2,
            "start_date": None
        }
    },
    {
        "name": "缺少file_path参数",
        "data": {
            "lookback": 100,
            "pred_len": 30,
            "temperature": 1.3,
            "top_p": 0.98,
            "sample_count": 2
        }
    },
    {
        "name": "无效股票代码",
        "data": {
            "file_path": "stock_invalid_live",
            "lookback": 100,
            "pred_len": 30,
            "temperature": 1.3,
            "top_p": 0.98,
            "sample_count": 2
        }
    },
    {
        "name": "lookback参数过大",
        "data": {
            "file_path": "stock_600519_live",
            "lookback": 10000,
            "pred_len": 30,
            "temperature": 1.3,
            "top_p": 0.98,
            "sample_count": 2
        }
    },
    {
        "name": "pred_len参数过大",
        "data": {
            "file_path": "stock_600519_live",
            "lookback": 100,
            "pred_len": 1000,
            "temperature": 1.3,
            "top_p": 0.98,
            "sample_count": 2
        }
    }
]

def test_predict_api():
    print("=== 预测API 400错误诊断测试 ===\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"测试 {i}: {test_case['name']}")
        print(f"请求参数: {json.dumps(test_case['data'], ensure_ascii=False, indent=2)}")
        
        try:
            response = requests.post(url, headers=headers, json=test_case['data'], timeout=30)
            
            print(f"状态码: {response.status_code}")
            
            if response.status_code == 200:
                print("✅ 请求成功")
                result = response.json()
                if 'error' in result:
                    print(f"响应错误: {result['error']}")
                else:
                    print(f"预测结果: 生成 {len(result.get('prediction_data', []))} 个预测点")
            elif response.status_code == 400:
                print("❌ 400错误 - 请求参数错误")
                try:
                    error_data = response.json()
                    print(f"错误详情: {error_data}")
                except:
                    print(f"错误响应文本: {response.text}")
            elif response.status_code == 500:
                print("❌ 500错误 - 服务器内部错误")
                try:
                    error_data = response.json()
                    print(f"错误详情: {error_data}")
                except:
                    print(f"错误响应文本: {response.text}")
            else:
                print(f"其他状态码: {response.status_code}")
                print(f"响应文本: {response.text}")
                
        except requests.exceptions.Timeout:
            print("❌ 请求超时")
        except requests.exceptions.ConnectionError:
            print("❌ 连接错误 - 服务器可能未启动")
        except Exception as e:
            print(f"❌ 请求异常: {e}")
        
        print("-" * 80)

if __name__ == "__main__":
    test_predict_api()