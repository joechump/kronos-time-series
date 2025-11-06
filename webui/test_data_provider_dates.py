#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据提供者日期参数处理
"""

import sys
import os
import pandas as pd
import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_data_provider_date_params():
    """测试数据提供者日期参数处理"""
    print("=" * 60)
    print("测试数据提供者日期参数处理")
    print("=" * 60)
    
    try:
        from akshare_data_provider import AkshareDataProvider
        
        # 创建数据提供者实例
        provider = AkshareDataProvider()
        
        # 测试日期参数
        end_date = datetime.datetime.now().strftime('%Y%m%d')
        start_date = (datetime.datetime.now() - datetime.timedelta(days=1825)).strftime('%Y%m%d')  # 5年
        
        print(f"测试日期范围: {start_date} 到 {end_date}")
        
        # 测试获取股票数据
        result = provider.get_stock_data('600523', 'daily', start_date, end_date, save_to_temp_file=False)
        
        # 检查返回结果类型
        if isinstance(result, tuple) and len(result) == 2:
            stock_data, temp_file_path = result
        else:
            # 兼容旧版本，只有股票数据
            stock_data = result
            temp_file_path = None
        
        if stock_data is not None and not stock_data.empty:
            print(f"✓ 成功获取股票数据，数据量: {len(stock_data)}")
            print(f"  列名: {list(stock_data.columns)}")
            print(f"  数据范围: {stock_data['date'].min()} 到 {stock_data['date'].max()}")
            if temp_file_path:
                print(f"  临时文件路径: {temp_file_path}")
            
            # 检查实际获取的数据天数
            actual_days = (stock_data['date'].max() - stock_data['date'].min()).days
            print(f"  实际数据天数: {actual_days} 天")
            
            # 检查数据行数
            print(f"  数据行数: {len(stock_data)} 行")
            
            # 检查是否有足够的数据进行预测
            if len(stock_data) >= 400:
                print("✓ 数据量足够进行预测 (≥400行)")
            else:
                print(f"⚠ 数据量不足，只有 {len(stock_data)} 行，需要至少 400 行")
                
        else:
            print("✗ 获取股票数据失败")
            
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_direct_akshare():
    """直接测试akshare获取数据"""
    print("\n" + "=" * 60)
    print("直接测试akshare获取数据")
    print("=" * 60)
    
    try:
        import akshare as ak
        
        end_date = datetime.datetime.now().strftime('%Y%m%d')
        start_date = (datetime.datetime.now() - datetime.timedelta(days=1825)).strftime('%Y%m%d')  # 5年
        
        print(f"直接测试akshare，日期范围: {start_date} 到 {end_date}")
        
        stock_data = ak.stock_zh_a_hist(symbol='600523', period='daily', start_date=start_date, end_date=end_date, adjust='')
        
        if stock_data is not None and not stock_data.empty:
            print(f"✓ 直接akshare获取成功，数据量: {len(stock_data)}")
            print(f"  列名: {list(stock_data.columns)}")
            print(f"  数据范围: {stock_data['日期'].min()} 到 {stock_data['日期'].max()}")
            print(f"  数据行数: {len(stock_data)} 行")
        else:
            print("✗ 直接akshare获取失败")
            
    except Exception as e:
        print(f"✗ 直接akshare测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_data_provider_date_params()
    test_direct_akshare()