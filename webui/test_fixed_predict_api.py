#!/usr/bin/env python3
"""
测试修复后的预测API
"""

import requests
import json

def test_predict_api():
    """测试预测API"""
    
    # 测试参数
    test_data = {
        "file_path": "stock_600519_live",
        "lookback": 400,
        "pred_len": 120,
        "temperature": 1.3,
        "top_p": 1,
        "sample_count": 1,
        "trading_mode": "calendar",
        "start_date": ""
    }
    
    print("🚀 开始测试预测API...")
    print(f"测试参数: {json.dumps(test_data, indent=2)}")
    
    try:
        # 发送预测请求
        response = requests.post(
            "http://localhost:7070/api/predict",
            json=test_data,
            timeout=60
        )
        
        print(f"\n📊 响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 预测API测试成功！")
            print(f"预测结果包含: {list(result.keys())}")
            
            if 'prediction' in result:
                pred_data = result['prediction']
                print(f"预测数据量: {len(pred_data)}")
                if len(pred_data) > 0:
                    print(f"第一个预测点: {pred_data[0]}")
            
            if 'chart' in result:
                print("✅ 图表数据生成成功")
            
            if 'stats' in result:
                print(f"统计信息: {result['stats']}")
                
        elif response.status_code == 400:
            error_data = response.json()
            print(f"❌ 预测API返回400错误: {error_data}")
            return False
            
        elif response.status_code == 500:
            error_data = response.json()
            print(f"❌ 预测API返回500错误: {error_data}")
            return False
            
        else:
            print(f"❌ 预测API返回未知错误: {response.status_code}")
            print(f"响应内容: {response.text}")
            return False
            
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务器，请确保Flask服务器正在运行")
        return False
        
    except requests.exceptions.Timeout:
        print("❌ 请求超时，预测可能需要更长时间")
        return False
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        return False

if __name__ == "__main__":
    success = test_predict_api()
    if success:
        print("\n🎉 预测API修复成功！")
    else:
        print("\n⚠️ 预测API仍存在问题，需要进一步调试")