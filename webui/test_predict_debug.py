#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预测API调试脚本 - 模拟前端预测请求，获取详细错误信息
"""

import requests
import json
import sys

def test_predict_api():
    """测试预测API接口"""
    
    # 模拟前端预测请求参数
    prediction_params = {
        "file_path": "stock_600519_live",
        "lookback": 400,
        "pred_len": 120,
        "temperature": 0.1,
        "top_p": 0.9,
        "sample_count": 1
    }
    
    print("=== 预测API调试测试 ===")
    print(f"请求参数: {json.dumps(prediction_params, indent=2, ensure_ascii=False)}")
    
    try:
        # 发送预测请求
        response = requests.post(
            "http://localhost:7070/api/predict",
            json=prediction_params,
            timeout=30
        )
        
        print(f"\n=== 响应信息 ===")
        print(f"状态码: {response.status_code}")
        print(f"响应头: {dict(response.headers)}")
        
        if response.status_code == 200:
            print("✅ 预测请求成功！")
            result = response.json()
            print(f"响应内容: {json.dumps(result, indent=2, ensure_ascii=False)}")
        else:
            print(f"❌ 预测请求失败，状态码: {response.status_code}")
            try:
                error_info = response.json()
                print(f"错误信息: {json.dumps(error_info, indent=2, ensure_ascii=False)}")
            except:
                print(f"原始响应内容: {response.text}")
                
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求异常: {e}")
    except Exception as e:
        print(f"❌ 其他异常: {e}")

def test_model_status():
    """测试模型状态接口"""
    
    print("\n=== 模型状态测试 ===")
    
    try:
        response = requests.get("http://localhost:7070/api/model-status", timeout=10)
        
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            status = response.json()
            print(f"模型状态: {json.dumps(status, indent=2, ensure_ascii=False)}")
        else:
            print(f"模型状态请求失败: {response.text}")
            
    except Exception as e:
        print(f"模型状态请求异常: {e}")

if __name__ == "__main__":
    # 先测试模型状态
    test_model_status()
    
    # 再测试预测API
    test_predict_api()