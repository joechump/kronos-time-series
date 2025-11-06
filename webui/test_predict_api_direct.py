#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接测试预测API的脚本
"""

import requests
import json
import sys
import os

def test_predict_api():
    """测试预测API"""
    print("=" * 60)
    print("直接测试预测API")
    print("=" * 60)
    
    # 模拟前端发送的预测请求
    url = "http://localhost:7070/api/predict"
    
    # 构建请求数据（模拟前端发送的数据）
    data = {
        "file_path": "stock_600519_live",
        "lookback": 400,
        "pred_len": 30,
        "temperature": 0.7,
        "top_p": 0.9,
        "sample_count": 1,
        "trading_mode": "calendar",
        "start_date": ""
    }
    
    print("1. 发送预测请求...")
    print(f"   请求URL: {url}")
    print(f"   请求数据: {json.dumps(data, indent=2, ensure_ascii=False)}")
    
    try:
        # 发送POST请求
        response = requests.post(url, json=data, timeout=30)
        
        print(f"\n2. 服务器响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            print("✓ 预测请求成功")
            result = response.json()
            print(f"   响应数据: {json.dumps(result, indent=2, ensure_ascii=False)}")
        else:
            print(f"✗ 预测请求失败，状态码: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   错误信息: {json.dumps(error_data, indent=2, ensure_ascii=False)}")
            except:
                print(f"   原始响应内容: {response.text}")
                
    except requests.exceptions.ConnectionError:
        print("✗ 无法连接到服务器，请确保服务器正在运行")
    except requests.exceptions.Timeout:
        print("✗ 请求超时，服务器响应时间过长")
    except Exception as e:
        print(f"✗ 请求异常: {e}")
    
    print("\n" + "=" * 60)
    print("预测API测试完成")
    print("=" * 60)

def test_model_status():
    """测试模型状态API"""
    print("\n" + "=" * 60)
    print("测试模型状态API")
    print("=" * 60)
    
    url = "http://localhost:7070/api/model-status"
    
    print("1. 发送模型状态请求...")
    print(f"   请求URL: {url}")
    
    try:
        # 发送GET请求
        response = requests.get(url, timeout=10)
        
        print(f"\n2. 服务器响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            print("✓ 模型状态请求成功")
            result = response.json()
            print(f"   响应数据: {json.dumps(result, indent=2, ensure_ascii=False)}")
        else:
            print(f"✗ 模型状态请求失败，状态码: {response.status_code}")
            print(f"   原始响应内容: {response.text}")
                
    except requests.exceptions.ConnectionError:
        print("✗ 无法连接到服务器，请确保服务器正在运行")
    except requests.exceptions.Timeout:
        print("✗ 请求超时，服务器响应时间过长")
    except Exception as e:
        print(f"✗ 请求异常: {e}")
    
    print("\n" + "=" * 60)
    print("模型状态API测试完成")
    print("=" * 60)

if __name__ == "__main__":
    # 先测试模型状态
    test_model_status()
    
    # 再测试预测API
    test_predict_api()