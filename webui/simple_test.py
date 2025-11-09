#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单测试脚本，直接测试预测API
"""

import requests
import json
import time

def test_predict_api():
    """测试预测API"""
    print("开始测试预测API...")
    
    # 等待服务器完全启动
    time.sleep(2)
    
    # 模拟前端发送的预测请求参数
    prediction_params = {
        "file_path": "stock_600519_live",
        "lookback": 400,
        "pred_len": 120,
        "temperature": 1.3,
        "top_p": 1,
        "sample_count": 1,
        "trading_mode": "calendar",
        "start_date": "2024-01-01"  # 使用修复后的YYYY-MM-DD格式
    }
    
    url = "http://localhost:8080/api/predict"
    
    print(f"请求URL: {url}")
    print(f"请求参数: {json.dumps(prediction_params, indent=2, ensure_ascii=False)}")
    
    try:
        response = requests.post(url, json=prediction_params, timeout=60)
        print(f"\n状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 预测API请求成功！")
            print(f"预测类型: {result.get('prediction_type', 'N/A')}")
            print(f"消息: {result.get('message', 'N/A')}")
            return True
        else:
            print(f"❌ 预测API返回错误: {response.status_code}")
            
            # 尝试获取详细的错误信息
            try:
                error_data = response.json()
                print(f"错误信息: {json.dumps(error_data, indent=2, ensure_ascii=False)}")
                
                # 分析错误类型
                if response.status_code == 400:
                    print("错误类型: 客户端错误 (400)")
                    if "模型未加载" in error_data.get('error', ''):
                        print("具体原因: Kronos模型未加载")
                    elif "数据长度不足" in error_data.get('error', ''):
                        print("具体原因: 数据量不足")
                    elif "无效的开始日期格式" in error_data.get('error', ''):
                        print("具体原因: 日期格式错误")
                    elif "文件路径不能为空" in error_data.get('error', ''):
                        print("具体原因: 文件路径为空")
                    else:
                        print("具体原因: 其他参数错误")
                        
            except:
                print(f"原始错误响应: {response.text}")
            
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务器，请确保Web服务器正在运行")
        return False
    except requests.exceptions.Timeout:
        print("❌ 请求超时，服务器可能正在处理大量数据")
        return False
    except Exception as e:
        print(f"❌ 请求异常: {e}")
        return False

if __name__ == "__main__":
    test_predict_api()