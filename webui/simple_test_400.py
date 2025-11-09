import requests
import json

def test_predict():
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
    
    print("测试预测API...")
    print(f"参数: {params}")
    
    try:
        response = requests.post(url, json=params, timeout=30)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ 请求成功!")
            result = response.json()
            print(f"响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
        else:
            print(f"❌ 请求失败")
            print(f"响应内容: {response.text}")
            
    except Exception as e:
        print(f"❌ 异常: {e}")

if __name__ == "__main__":
    test_predict()