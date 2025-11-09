#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查实际的股票数据情况
"""

import requests
import json
import pandas as pd
from datetime import datetime

def check_akshare_api():
    """直接测试akshare数据接口"""
    print("=== 直接测试akshare数据接口 ===")
    
    # 测试不同的股票代码
    test_symbols = ['000001', '000002', '600000', '000858']
    
    for symbol in test_symbols:
        print(f"\n测试股票代码: {symbol}")
        
        try:
            response = requests.post(
                'http://localhost:8080/api/akshare/get-stock-data',
                json={
                    'symbol': symbol,
                    'period': 'daily',
                    'start_date': '20240101',
                    'end_date': '20251231'
                },
                timeout=30
            )
            
            print(f"  状态码: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"  成功: {data.get('success')}")
                print(f"  数据量: {data.get('data_info', {}).get('rows', 0)}")
                
                if data.get('success') and 'data' in data:
                    stock_data = data['data']
                    if len(stock_data) > 0:
                        # 显示前5条和后5条数据
                        print(f"  前5条数据:")
                        for i, item in enumerate(stock_data[:5]):
                            print(f"    {i+1}. {item}")
                        
                        print(f"  后5条数据:")
                        for i, item in enumerate(stock_data[-5:]):
                            print(f"    {i+1}. {item}")
                        
                        # 提取日期范围
                        dates = [item.get('date') for item in stock_data if 'date' in item]
                        if dates:
                            min_date = min(dates)
                            max_date = max(dates)
                            print(f"  日期范围: {min_date} 到 {max_date}")
                            
                            # 检查2025年3月数据
                            march_2025 = [d for d in dates if d and '2025-03' in d]
                            print(f"  2025年3月数据天数: {len(march_2025)}")
                            
                            # 检查3月29日
                            has_0329 = any(d for d in dates if d and d.endswith('03-29'))
                            print(f"  是否有3月29日数据: {'是' if has_0329 else '否'}")
                    else:
                        print("  数据为空")
                else:
                    print(f"  错误信息: {data.get('error', '未知错误')}")
            else:
                print(f"  错误响应: {response.text}")
                
        except Exception as e:
            print(f"  请求异常: {e}")

def test_predict_api_with_real_data():
    """使用真实数据测试预测API"""
    print("\n=== 测试预测API（使用真实数据） ===")
    
    # 先获取股票数据来确定可用的日期范围
    try:
        response = requests.post(
            'http://localhost:8080/api/akshare/get-stock-data',
            json={
                'symbol': '000001',
                'period': 'daily',
                'start_date': '20240101',
                'end_date': '20251231'
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success') and 'data' in data and len(data['data']) > 0:
                stock_data = data['data']
                dates = [item.get('date') for item in stock_data if 'date' in item]
                
                if dates:
                    min_date = min(dates)
                    max_date = max(dates)
                    print(f"可用数据范围: {min_date} 到 {max_date}")
                    
                    # 选择测试日期
                    test_dates = []
                    
                    # 如果有2025年3月数据，测试边界日期
                    march_dates = [d for d in dates if '2025-03' in d]
                    if march_dates:
                        march_dates.sort()
                        print(f"2025年3月可用日期: {march_dates[:5]} ... {march_dates[-5:]}")
                        
                        # 测试3月28日、29日、31日
                        for test_date in ['2025-03-28', '2025-03-29', '2025-03-31']:
                            if test_date in dates:
                                test_dates.append(test_date)
                            else:
                                print(f"警告: {test_date} 不在数据中")
                    
                    # 如果没有3月数据，使用可用的日期
                    if not test_dates and len(dates) >= 40:  # 需要足够的数据
                        # 选择数据范围中间的日期
                        mid_idx = len(dates) // 2
                        test_date = dates[mid_idx]
                        test_dates.append(test_date)
                        print(f"使用中间日期测试: {test_date}")
                    
                    # 测试预测API
                    for test_date in test_dates:
                        print(f"\n测试预测日期: {test_date}")
                        
                        response = requests.post(
                            'http://localhost:8080/api/predict',
                            json={
                                'file_path': 'stock_000001_live',
                                'lookback': 30,
                                'pred_len': 10,
                                'start_date': test_date
                            },
                            timeout=30
                        )
                        
                        print(f"  状态码: {response.status_code}")
                        
                        if response.status_code == 200:
                            result = response.json()
                            print("  ✅ 预测成功")
                            if 'prediction' in result:
                                pred_data = result['prediction']
                                print(f"    预测天数: {len(pred_data)}")
                        else:
                            try:
                                error_data = response.json()
                                print(f"  ❌ 预测失败: {error_data.get('error', '未知错误')}")
                                if 'suggestion' in error_data:
                                    print(f"     建议: {error_data['suggestion']}")
                            except:
                                print(f"  ❌ 错误响应: {response.text}")
                else:
                    print("❌ 没有可用的日期数据")
            else:
                print("❌ 无法获取股票数据")
        else:
            print("❌ 股票数据接口请求失败")
            
    except Exception as e:
        print(f"❌ 测试异常: {e}")

def check_server_status():
    """检查服务器状态"""
    print("\n=== 检查服务器状态 ===")
    
    try:
        response = requests.get('http://localhost:8080/', timeout=10)
        print(f"主页状态: {response.status_code}")
        
        response = requests.get('http://localhost:8080/api/model-status', timeout=10)
        print(f"模型状态: {response.status_code}")
        if response.status_code == 200:
            status_data = response.json()
            print(f"  模型信息: {status_data}")
            
    except Exception as e:
        print(f"服务器检查异常: {e}")

def main():
    """主函数"""
    print("检查实际的股票数据情况")
    print("=" * 60)
    
    # 检查服务器状态
    check_server_status()
    
    # 测试akshare接口
    check_akshare_api()
    
    # 测试预测API
    test_predict_api_with_real_data()
    
    print("\n" + "=" * 60)
    print("检查完成")

if __name__ == "__main__":
    main()