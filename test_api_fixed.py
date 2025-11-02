import requests
import json

def test_predict_api():
    # API端点
    url = "http://localhost:7070/api/predict"
    
    # 预测参数
    payload = {
        "file_path": "stock_600519_live",  # 股票代码
        "lookback": 400,                   # 回看期数
        "pred_len": 120,                   # 预测期数
        "temperature": 1.0,                # 温度参数
        "top_p": 0.9,                      # Top-p采样
        "sample_count": 1                  # 采样次数
    }
    
    # 设置请求头
    headers = {
        'Content-Type': 'application/json'
    }
    
    try:
        # 发送POST请求
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        
        # 检查响应状态码
        if response.status_code == 200:
            print("✅ 预测请求成功!")
            print(f"响应状态码: {response.status_code}")
            print(f"响应长度: {len(response.content)} 字节")
            
            # 尝试解析JSON响应
            try:
                result = response.json()
                print(f"响应字段: {list(result.keys())}")
                
                # 检查是否有预测数据
                if 'prediction_data' in result:
                    print(f"预测数据点数量: {len(result['prediction_data'])}")
                if 'actual_data' in result:
                    print(f"实际数据点数量: {len(result['actual_data'])}")
                    
            except json.JSONDecodeError:
                print("无法解析JSON响应")
                print(f"响应内容前200字符: {response.text[:200]}")
        else:
            print(f"❌ 预测请求失败，状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求异常: {e}")

if __name__ == "__main__":
    test_predict_api()