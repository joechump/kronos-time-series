import requests
import json
from datetime import datetime, timedelta

def test_predict_api(params, description):
    print(f"\n{'='*50}")
    print(f"测试: {description}")
    print(f"{'='*50}")
    print("发送参数:")
    print(json.dumps(params, indent=2, ensure_ascii=False))
    
    try:
        response = requests.post(
            "http://localhost:7070/api/predict",
            json=params,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"\n响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ 请求成功!")
            try:
                response_data = response.json()
                print(f"响应消息: {response_data.get('message', '无消息')}")
            except:
                print("响应内容:", response.text[:200] + "..." if len(response.text) > 200 else response.text)
        elif response.status_code == 400:
            print("❌ 请求失败，收到400错误!")
            try:
                error_data = response.json()
                print(f"错误信息: {error_data.get('error', '无错误详情')}")
            except:
                print("响应内容:", response.text[:200] + "..." if len(response.text) > 200 else response.text)
        else:
            print(f"❌ 请求失败，收到{response.status_code}错误!")
            print("响应内容:", response.text[:200] + "..." if len(response.text) > 200 else response.text)
            
    except Exception as e:
        print(f"❌ 请求过程中发生异常: {e}")

# 基础测试参数
base_params = {
    "file_path": "stock_600519_live",
    "lookback": 100,
    "pred_len": 30,
    "temperature": 1.3,
    "top_p": 0.98,
    "sample_count": 2
}

# 测试1: 基础参数（不带start_date）
test_params_1 = base_params.copy()
test_predict_api(test_params_1, "基础参数（不带start_date）")

# 测试2: 带start_date参数
test_params_2 = base_params.copy()
test_params_2["start_date"] = "2023-01-01T00:00"
test_predict_api(test_params_2, "带start_date参数")

# 测试3: 带start_date参数（更近的时间）
test_params_3 = base_params.copy()
recent_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M")
test_params_3["start_date"] = recent_date
test_predict_api(test_params_3, "带start_date参数（最近时间）")

# 测试4: 缺少必需参数
test_params_4 = {
    "file_path": "stock_600519_live",
    # 缺少 lookback, pred_len
}
test_predict_api(test_params_4, "缺少必需参数")

# 测试5: 错误的file_path格式
test_params_5 = base_params.copy()
test_params_5["file_path"] = "invalid_path"
test_predict_api(test_params_5, "错误的file_path格式")

print(f"\n{'='*50}")
print("所有测试完成")
print(f"{'='*50}")