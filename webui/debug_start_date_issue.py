#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试start_date参数400错误问题
问题：当start_date选择在2025年3月28日之前时正常，3月29日之后出现400错误
"""

import requests
import json
import pandas as pd
from datetime import datetime, timedelta

def test_stock_data_range():
    """测试股票数据的实际日期范围"""
    print("=== 测试股票数据日期范围 ===")
    
    # 测试获取股票数据
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
            if data.get('success'):
                data_info = data.get('data_info', {})
                print(f"✅ 股票数据获取成功")
                print(f"  数据量: {data_info.get('rows', 0)} 条记录")
                print(f"  开始日期: {data_info.get('start_date', 'N/A')}")
                print(f"  结束日期: {data_info.get('end_date', 'N/A')}")
                
                # 如果有实际数据，显示日期范围
                if 'data' in data and len(data['data']) > 0:
                    dates = [item.get('date') for item in data['data'] if 'date' in item]
                    if dates:
                        min_date = min(dates)
                        max_date = max(dates)
                        print(f"  实际数据范围: {min_date} 到 {max_date}")
                        
                        # 检查2025年3月的数据
                        march_2025_dates = [d for d in dates if d and d.startswith('2025-03')]
                        print(f"  2025年3月数据天数: {len(march_2025_dates)}")
                        if march_2025_dates:
                            print(f"  3月最早日期: {min(march_2025_dates)}")
                            print(f"  3月最晚日期: {max(march_2025_dates)}")
                        
                        # 检查3月29日是否有数据
                        has_0329 = any(d for d in dates if d and d == '2025-03-29')
                        print(f"  3月29日是否有数据: {'是' if has_0329 else '否'}")
                        
                        return True, dates
            else:
                print(f"❌ 数据获取失败: {data.get('error', '未知错误')}")
        else:
            print(f"❌ HTTP错误: {response.status_code}")
            print(f"   响应内容: {response.text}")
            
    except Exception as e:
        print(f"❌ 测试异常: {e}")
    
    return False, []

def test_predict_api_with_dates():
    """测试预测API在不同日期下的表现"""
    print("\n=== 测试预测API日期边界 ===")
    
    # 测试日期序列
    test_dates = [
        '2025-03-25',  # 应该正常
        '2025-03-26',  # 应该正常
        '2025-03-27',  # 应该正常
        '2025-03-28',  # 应该正常
        '2025-03-29',  # 可能出错
        '2025-03-30',  # 周末，可能无数据
        '2025-03-31',  # 可能出错
        '2025-04-01',  # 可能出错
    ]
    
    for test_date in test_dates:
        print(f"\n测试日期: {test_date}")
        
        try:
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
                print("  ✅ 预测成功")
            else:
                try:
                    error_data = response.json()
                    print(f"  ❌ 预测失败: {error_data.get('error', '未知错误')}")
                    if 'suggestion' in error_data:
                        print(f"     建议: {error_data['suggestion']}")
                except:
                    print(f"  ❌ 响应内容: {response.text}")
                    
        except Exception as e:
            print(f"  ❌ 请求异常: {e}")

def analyze_data_sufficiency():
    """分析数据充足性问题"""
    print("\n=== 分析数据充足性 ===")
    
    # 获取当前日期
    today = datetime.now()
    
    # 计算不同开始日期需要的数据量
    lookback = 30
    pred_len = 10
    required_data = lookback + pred_len  # 40个数据点
    
    print(f"预测参数: lookback={lookback}, pred_len={pred_len}")
    print(f"需要数据点: {required_data}")
    
    # 测试不同开始日期的数据充足性
    test_cases = [
        ('2025-03-28', "临界点前"),
        ('2025-03-29', "临界点后"),
        ('2025-03-31', "月末"),
        ('2025-04-01', "下月初")
    ]
    
    for test_date, description in test_cases:
        print(f"\n分析 {test_date} ({description}):")
        
        # 计算从该日期开始需要的数据范围
        start_dt = pd.to_datetime(test_date)
        
        # 假设数据按交易日提供，需要计算实际需要的交易日数量
        # 通常股票市场一周有5个交易日
        trading_days_needed = required_data
        calendar_days_needed = trading_days_needed * 7 / 5  # 粗略估计
        
        print(f"  需要交易日数: {trading_days_needed}")
        print(f"  大约需要日历天数: {calendar_days_needed:.1f}")
        
        # 检查数据是否足够
        # 这里需要实际的股票数据来验证
        
        # 测试API响应
        try:
            response = requests.post(
                'http://localhost:8080/api/predict',
                json={
                    'file_path': 'stock_000001_live',
                    'lookback': lookback,
                    'pred_len': pred_len,
                    'start_date': test_date
                },
                timeout=30
            )
            
            if response.status_code == 400:
                try:
                    error_data = response.json()
                    error_msg = error_data.get('error', '')
                    if '数据不足' in error_msg:
                        print(f"  ❌ 数据不足: {error_msg}")
                    else:
                        print(f"  ❌ 其他错误: {error_msg}")
                except:
                    print(f"  ❌ 错误响应: {response.text}")
            else:
                print(f"  ✅ API响应正常")
                
        except Exception as e:
            print(f"  ❌ 请求异常: {e}")

def main():
    """主函数"""
    print("开始调试start_date参数400错误问题")
    print("=" * 60)
    
    # 测试股票数据范围
    success, dates = test_stock_data_range()
    
    if success:
        # 测试预测API
        test_predict_api_with_dates()
        
        # 分析数据充足性
        analyze_data_sufficiency()
    else:
        print("\n❌ 无法获取股票数据，无法继续测试")
    
    print("\n" + "=" * 60)
    print("调试完成")

if __name__ == "__main__":
    main()