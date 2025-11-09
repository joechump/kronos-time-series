#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查股票数据范围，了解为什么2025-03-29及之后的日期数据不足
"""

import requests
import json
import pandas as pd
from datetime import datetime, timedelta

def check_stock_data_range(symbol):
    """检查股票数据的时间范围"""
    url = "http://localhost:8080/api/akshare/get-stock-data"
    
    # 获取最近3年的数据
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=1095)).strftime('%Y%m%d')
    
    params = {
        'symbol': symbol,
        'period': 'daily',
        'start_date': start_date,
        'end_date': end_date
    }
    
    try:
        response = requests.post(url, json=params)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                stock_data = data.get('data', [])
                if stock_data:
                    # 提取日期并排序
                    dates = [pd.to_datetime(item['date']) for item in stock_data if 'date' in item]
                    dates_sorted = sorted(dates)
                    
                    print(f"📊 股票 {symbol} 数据范围:")
                    print(f"   最早日期: {dates_sorted[0].strftime('%Y-%m-%d')}")
                    print(f"   最晚日期: {dates_sorted[-1].strftime('%Y-%m-%d')}")
                    print(f"   总数据量: {len(dates_sorted)} 条记录")
                    
                    # 检查特定日期的数据可用性
                    test_dates = [
                        '2025-03-27',
                        '2025-03-28', 
                        '2025-03-29',
                        '2025-03-30',
                        '2025-03-31',
                        '2025-04-01',
                        '2025-04-02'
                    ]
                    
                    print(f"\n📅 特定日期数据可用性检查:")
                    for test_date in test_dates:
                        test_dt = pd.to_datetime(test_date)
                        # 计算从该日期开始的数据量
                        data_after_date = [d for d in dates_sorted if d >= test_dt]
                        print(f"   {test_date}: {len(data_after_date)} 个数据点")
                    
                    return dates_sorted
                else:
                    print(f"❌ 股票 {symbol} 无数据")
                    return None
            else:
                print(f"❌ 股票 {symbol} 数据获取失败: {data.get('error', '未知错误')}")
                return None
        else:
            print(f"❌ API请求失败，状态码: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ 检查股票数据范围异常: {e}")
        return None

def analyze_data_sufficiency():
    """分析数据充足性问题"""
    print("🚀 分析股票数据充足性问题")
    print("=" * 60)
    
    # 测试多个股票
    symbols = ['000858', '600519', '000001']
    
    for symbol in symbols:
        print(f"\n📈 分析股票 {symbol}:")
        print("-" * 40)
        
        dates = check_stock_data_range(symbol)
        if dates:
            # 分析预测需求
            lookback = 120
            pred_len = 30
            required = lookback + pred_len
            
            print(f"\n📊 预测需求分析:")
            print(f"   lookback: {lookback}")
            print(f"   pred_len: {pred_len}")
            print(f"   需要数据点: {required}")
            
            # 找到临界日期
            critical_date = None
            for i in range(len(dates)):
                if len(dates) - i < required:
                    critical_date = dates[i]
                    break
            
            if critical_date:
                print(f"⚠️  临界日期: {critical_date.strftime('%Y-%m-%d')}")
                print(f"   从该日期开始数据不足 {required} 个点")
            else:
                print("✅ 所有日期数据充足")
                
            # 检查2025-03-29及之后的日期
            march_29 = pd.to_datetime('2025-03-29')
            data_after_march_29 = [d for d in dates if d >= march_29]
            print(f"\n📅 2025-03-29及之后的数据量: {len(data_after_march_29)}")
            
            if len(data_after_march_29) < required:
                print(f"❌ 2025-03-29及之后的数据不足，需要 {required} 个，实际只有 {len(data_after_march_29)} 个")
            else:
                print("✅ 2025-03-29及之后的数据充足")

def main():
    """主函数"""
    analyze_data_sufficiency()
    
    print("\n" + "=" * 60)
    print("💡 建议:")
    print("1. 如果数据确实不足，需要更新股票数据到更晚的日期")
    print("2. 或者调整预测参数，减少lookback或pred_len的值")
    print("3. 或者选择更早的开始日期进行预测")

if __name__ == "__main__":
    main()