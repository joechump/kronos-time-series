#!/usr/bin/env python3
"""
测试股票代码600525的预测API
"""

import requests
import json

# 测试股票代码600525的预测API
try:
    url = 'http://localhost:7070/api/predict'
    data = {
        'file_path': 'stock_600525_live',
        'lookback': 400,
        'pred_len': 120,
        'model_name': 'kronos-small'
    }
    
    print('测试API调用...')
    print(f'请求参数: {data}')
    
    response = requests.post(url, json=data)
    print(f'状态码: {response.status_code}')
    
    if response.status_code == 200:
        result = response.json()
        print(f'API调用成功: {result.get("success", "Unknown")}')
        if 'predictions' in result:
            print(f'预测结果数量: {len(result["predictions"])}')
    else:
        print(f'API调用失败: {response.text}')
        
except Exception as e:
    print(f'请求异常: {e}')