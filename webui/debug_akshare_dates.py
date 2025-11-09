#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试akshare数据提供者的日期排序问题
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from akshare_data_provider import AkshareDataProvider
import pandas as pd
from datetime import datetime, timedelta

def test_akshare_direct():
    """直接测试akshare库获取数据"""
    print("=== 直接测试akshare库 ===")
    
    try:
        import akshare as ak
        
        # 测试不同的股票代码
        test_stocks = ['000001', '600519', '000858']
        
        for stock_code in test_stocks:
            print(f"\n--- 测试股票 {stock_code} ---")
            
            # 设置日期范围
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')  # 1年
            
            print(f"日期范围: {start_date} 到 {end_date}")
            
            try:
                # 直接调用akshare
                stock_data = ak.stock_zh_a_hist(
                    symbol=stock_code, 
                    period='daily', 
                    start_date=start_date, 
                    end_date=end_date,
                    adjust="hfq"
                )
                
                if stock_data is not None and not stock_data.empty:
                    print(f"✅ 直接akshare获取成功")
                    print(f"  原始数据形状: {stock_data.shape}")
                    print(f"  原始列名: {list(stock_data.columns)}")
                    
                    # 显示前5行和后5行
                    print(f"  前5行:")
                    print(stock_data.head())
                    print(f"  后5行:")
                    print(stock_data.tail())
                    
                    # 检查日期列
                    if '日期' in stock_data.columns:
                        dates = stock_data['日期'].tolist()
                        print(f"  日期范围: {dates[0]} 到 {dates[-1]}")
                        
                        # 检查排序
                        is_sorted = all(dates[i] <= dates[i+1] for i in range(len(dates)-1))
                        print(f"  日期是否排序: {'是' if is_sorted else '否'}")
                        
                        if not is_sorted:
                            print(f"  ❌ 日期排序有问题!")
                            # 显示排序前后的对比
                            sorted_dates = sorted(dates)
                            print(f"  排序前: {dates[:3]} ... {dates[-3:]}")
                            print(f"  排序后: {sorted_dates[:3]} ... {sorted_dates[-3:]}")
                    
                else:
                    print(f"❌ 直接akshare获取失败，数据为空")
                    
            except Exception as e:
                print(f"❌ 直接akshare调用异常: {e}")
                
    except ImportError:
        print("❌ akshare库未安装")

def test_data_provider():
    """测试数据提供者"""
    print("\n=== 测试数据提供者 ===")
    
    provider = AkshareDataProvider()
    
    # 测试股票代码
    test_stocks = ['000001', '600519', '000858']
    
    for stock_code in test_stocks:
        print(f"\n--- 测试股票 {stock_code} ---")
        
        # 设置日期范围
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')  # 1年
        
        print(f"日期范围: {start_date} 到 {end_date}")
        
        try:
            # 使用数据提供者
            stock_data = provider.get_stock_data(stock_code, 'daily', start_date, end_date)
            
            if stock_data is not None and not stock_data.empty:
                print(f"✅ 数据提供者获取成功")
                print(f"  数据形状: {stock_data.shape}")
                print(f"  列名: {list(stock_data.columns)}")
                
                # 显示前5行和后5行
                print(f"  前5行:")
                print(stock_data.head())
                print(f"  后5行:")
                print(stock_data.tail())
                
                # 检查日期列
                if 'date' in stock_data.columns:
                    dates = stock_data['date'].tolist()
                    print(f"  日期范围: {dates[0]} 到 {dates[-1]}")
                    
                    # 检查排序
                    is_sorted = all(dates[i] <= dates[i+1] for i in range(len(dates)-1))
                    print(f"  日期是否排序: {'是' if is_sorted else '否'}")
                    
                    if not is_sorted:
                        print(f"  ❌ 日期排序有问题!")
                        # 显示排序前后的对比
                        sorted_dates = sorted(dates)
                        print(f"  排序前: {dates[:3]} ... {dates[-3:]}")
                        print(f"  排序后: {sorted_dates[:3]} ... {sorted_dates[-3:]}")
                        
                    # 检查2025年3月数据
                    march_2025_dates = [d for d in dates if d.year == 2025 and d.month == 3]
                    print(f"  2025年3月数据天数: {len(march_2025_dates)}")
                    if march_2025_dates:
                        print(f"  3月最早日期: {min(march_2025_dates)}")
                        print(f"  3月最晚日期: {max(march_2025_dates)}")
                        
                        # 检查3月29日
                        has_0329 = any(d for d in march_2025_dates if d.day == 29)
                        print(f"  是否有3月29日数据: {'是' if has_0329 else '否'}")
                
            else:
                print(f"❌ 数据提供者获取失败，数据为空")
                
        except Exception as e:
            print(f"❌ 数据提供者调用异常: {e}")

def test_api_data():
    """测试API返回的数据"""
    print("\n=== 测试API返回的数据 ===")
    
    import requests
    import json
    
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
            print(f"✅ API请求成功")
            
            if data.get('success') and 'data' in data:
                stock_data = data['data']
                print(f"  数据量: {len(stock_data)}")
                
                if len(stock_data) > 0:
                    # 提取日期
                    dates = [item.get('date') for item in stock_data if 'date' in item]
                    print(f"  日期数量: {len(dates)}")
                    
                    if dates:
                        # 转换日期格式
                        try:
                            date_objs = [pd.to_datetime(d) for d in dates]
                            min_date = min(date_objs)
                            max_date = max(date_objs)
                            print(f"  日期范围: {min_date} 到 {max_date}")
                            
                            # 检查排序
                            is_sorted = all(date_objs[i] <= date_objs[i+1] for i in range(len(date_objs)-1))
                            print(f"  日期是否排序: {'是' if is_sorted else '否'}")
                            
                            if not is_sorted:
                                print(f"  ❌ API返回数据日期排序有问题!")
                                
                            # 检查2025年3月数据
                            march_2025_dates = [d for d in date_objs if d.year == 2025 and d.month == 3]
                            print(f"  2025年3月数据天数: {len(march_2025_dates)}")
                            
                        except Exception as e:
                            print(f"  日期处理异常: {e}")
                            
            else:
                print(f"❌ API返回数据失败: {data.get('error', '未知错误')}")
        else:
            print(f"❌ API请求失败: {response.status_code}")
            
    except Exception as e:
        print(f"❌ API测试异常: {e}")

def main():
    """主函数"""
    print("调试akshare数据提供者的日期排序问题")
    print("=" * 60)
    
    # 测试直接akshare
    test_akshare_direct()
    
    # 测试数据提供者
    test_data_provider()
    
    # 测试API数据
    test_api_data()
    
    print("\n" + "=" * 60)
    print("调试完成")

if __name__ == "__main__":
    main()