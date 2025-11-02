import requests
import json

# 模拟前端发送的请求参数
prediction_params = {
    "file_path": "stock_600519_live",
    "lookback": 100,
    "pred_len": 30,
    "start_date": "2023-01-01T00:00",
    "temperature": 1.3,
    "top_p": 0.98,
    "sample_count": 2
}

print("发送预测请求参数:")
print(json.dumps(prediction_params, indent=2, ensure_ascii=False))

# 发送POST请求到API端点
try:
    response = requests.post(
        "http://localhost:7070/api/predict",
        json=prediction_params,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"\n响应状态码: {response.status_code}")
    print(f"响应内容: {response.text}")
    
    if response.status_code == 200:
        print("\n✅ 请求成功!")
        response_data = response.json()
        print(json.dumps(response_data, indent=2, ensure_ascii=False))
    elif response.status_code == 400:
        print("\n❌ 请求失败，收到400错误!")
        try:
            error_data = response.json()
            print(f"错误信息: {error_data}")
        except:
            print("无法解析错误响应")
    else:
        print(f"\n❌ 请求失败，收到{response.status_code}错误!")
        
except Exception as e:
    print(f"\n❌ 请求过程中发生异常: {e}")