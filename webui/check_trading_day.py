#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查特定日期是否为交易日
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from akshare_data_provider import AkshareDataProvider
import pandas as pd
from datetime import datetime

def check_trading_days():
    """检查特定日期是否为交易日"""
    print("=== 检查交易日情况 ===")
    
    provider = AkshareDataProvider()
    
    # 检查2025年3月的交易日情况
    test_dates = [
        '2025-03-24', '2025-03-25', '2025-03-26', '2025-03-27', '2025-03-28',
        '2025-03-29', '2025-03-30', '2025-03-31', '2025-04-01'
    ]
    
    for date_str in test_dates:
        # 转换为YYYYMMDD格式
        date_dt = datetime.strptime(date_str, '%Y-%m-%d')
        date_ymd = date_dt.strftime('%Y%m%d')
        
        # 检查是否为交易日
        is_trading = provider.is_trading_day(date_ymd)
        print(f"{date_str} ({date_dt.strftime('%A')}): {'✅ 交易日' if is_trading else '❌ 非交易日'}")

def check_march_2025_calendar():
    """检查2025年3月的交易日历"""
    print("\n=== 检查2025年3月交易日历 ===")
    
    provider = AkshareDataProvider()
    
    try:
        # 获取2025年3月的交易日历
        calendar = provider.get_trading_calendar('20250301', '20250331')
        
        if calendar is not None and not calendar.empty:
            trading_days = calendar['trade_date'].tolist()
            trading_days_str = [d.strftime('%Y-%m-%d') for d in trading_days]
            
            print(f"2025年3月交易日数量: {len(trading_days)}")
            print(f"交易日列表: {trading_days_str}")
            
            # 检查3月29日
            has_0329 = any('2025-03-29' in d for d in trading_days_str)
            print(f"3月29日是否为交易日: {'是' if has_0329 else '否'}")
            
            # 检查3月28日和3月31日
            has_0328 = any('2025-03-28' in d for d in trading_days_str)
            has_0331 = any('2025-03-31' in d for d in trading_days_str)
            print(f"3月28日是否为交易日: {'是' if has_0328 else '否'}")
            print(f"3月31日是否为交易日: {'是' if has_0331 else '否'}")
            
        else:
            print("❌ 无法获取交易日历")
            
    except Exception as e:
        print(f"❌ 获取交易日历异常: {e}")

def test_predict_api_with_real_scenario():
    """测试真实场景下的预测API"""
    print("\n=== 测试真实场景预测API ===")
    
    import requests
    import json
    
    # 测试场景：从3月28日开始预测 vs 从3月31日开始预测
    test_cases = [
        ('2025-03-28', "3月28日（周五）"),
        ('2025-03-31', "3月31日（周一）"),
        ('2025-04-01', "4月1日（周二）")
    ]
    
    for test_date, description in test_cases:
        print(f"\n测试 {description}:")
        
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
                result = response.json()
                print("  ✅ 预测成功")
                if 'prediction' in result:
                    pred_data = result['prediction']
                    print(f"    预测天数: {len(pred_data)}")
            else:
                try:
                    error_data = response.json()
                    error_msg = error_data.get('error', '未知错误')
                    print(f"  ❌ 预测失败: {error_msg}")
                    
                    # 分析错误类型
                    if '数据不足' in error_msg:
                        print("  💡 错误类型: 数据不足")
                        if 'suggestion' in error_data:
                            print(f"     建议: {error_data['suggestion']}")
                    elif '超出允许范围' in error_msg:
                        print("  💡 错误类型: 日期超出范围")
                    elif '无效的开始日期格式' in error_msg:
                        print("  💡 错误类型: 日期格式错误")
                    else:
                        print("  💡 错误类型: 其他错误")
                        
                except:
                    print(f"  ❌ 错误响应: {response.text}")
                    
        except Exception as e:
            print(f"  ❌ 请求异常: {e}")

def analyze_data_sufficiency_for_dates():
    """分析特定日期的数据充足性"""
    print("\n=== 分析数据充足性 ===")
    
    import requests
    
    # 获取股票数据
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
            if data.get('success') and 'data' in data:
                stock_data = data['data']
                
                # 提取日期
                dates = [item.get('date') for item in stock_data if 'date' in item]
                dates_sorted = sorted([pd.to_datetime(d) for d in dates])
                
                print(f"总数据量: {len(dates_sorted)}")
                print(f"数据范围: {dates_sorted[0]} 到 {dates_sorted[-1]}")
                
                # 分析3月28日之后的数据量
                march_28 = pd.to_datetime('2025-03-28')
                data_after_0328 = [d for d in dates_sorted if d >= march_28]
                
                print(f"3月28日之后的数据量: {len(data_after_0328)}")
                
                # 分析3月31日之后的数据量
                march_31 = pd.to_datetime('2025-03-31')
                data_after_0331 = [d for d in dates_sorted if d >= march_31]
                
                print(f"3月31日之后的数据量: {len(data_after_0331)}")
                
                # 预测需要的数据量
                lookback = 30
                pred_len = 10
                required = lookback + pred_len
                
                print(f"预测需要的数据量: {required}")
                
                # 检查充足性
                print(f"3月28日数据是否充足: {'是' if len(data_after_0328) >= required else '否'}")
                print(f"3月31日数据是否充足: {'是' if len(data_after_0331) >= required else '否'}")
                
        else:
            print("❌ 无法获取股票数据")
            
    except Exception as e:
        print(f"❌ 分析异常: {e}")

def main():
    """主函数"""
    print("检查特定日期是否为交易日及数据充足性")
    print("=" * 60)
    
    # 检查交易日
    check_trading_days()
    
    # 检查交易日历
    check_march_2025_calendar()
    
    # 分析数据充足性
    analyze_data_sufficiency_for_dates()
    
    # 测试预测API
    test_predict_api_with_real_scenario()
    
    print("\n" + "=" * 60)
    print("检查完成")

if __name__ == "__main__":
    main()