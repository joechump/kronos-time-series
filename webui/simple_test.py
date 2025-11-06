#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单测试修复效果
"""

import requests
import json

def test_predict_api():
    """测试预测API"""
    print("测试预测API...")
    
    try:
        url = 'http://localhost:7070/api/predict'
        data = {
            'stock_code': '600523',
            'pred_len': 30,
            'lookback': 100
        }
        
        response = requests.post(url, json=data, timeout=30)
        print(f'状态码: {response.status_code}')
        
        if response.status_code == 200:
            result = response.json()
            print('预测成功!')
            print(f'数据量: {result.get("data_length", 0)}')
            print(f'使用的lookback: {result.get("lookback_used", 0)}')
            print(f'预测结果数量: {len(result.get("predictions", []))}')
            return True
        else:
            print(f'错误: {response.text}')
            return False
            
    except requests.exceptions.ConnectionError:
        print('Web服务器未启动，请先启动服务器')
        return False
    except Exception as e:
        print(f'请求失败: {e}')
        return False

if __name__ == "__main__":
    test_predict_api()