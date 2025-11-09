#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单预测API测试
"""

import requests
import json

def test_predict():
    """测试预测API"""
    
    # 模拟前端请求参数
    params = {
        "file_path": "stock_600519_live",
        "lookback": 100,
        "pred_len": 30,
        "start_date": "2024-01-01",
        "temperature": 1.3,
        "top_p": 0.98,
        "sample_count": 2
    }
    
    print("测试预测API...")
    print("参数:", json.dumps(params, indent=2, ensure_ascii=False))
    
    try:
        response = requests.post("http://127.0.0.1:8080/api/predict", json=params, timeout=30)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            print("成功!")
            result = response.json()
            print("响应:", json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("失败!")
            print("错误响应:", response.text)
            
    except Exception as e:
        print(f"异常: {e}")

if __name__ == "__main__":
    test_predict()