#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据提供者日期参数处理 - 调试版本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from akshare_data_provider import AkshareDataProvider
import pandas as pd
from datetime import datetime, timedelta

def test_data_provider_date_params():
    """测试数据提供者日期参数处理"""
    print("=" * 60)
    print("测试数据提供者日期参数处理 - 调试版本")
    print("=" * 60)
    
    # 创建数据提供者实例
    data_provider = AkshareDataProvider()
    
    # 测试股票代码
    stock_code = "600523"
    
    # 测试1: 使用5年日期范围（1825天）
    print("\n测试1: 使用5年日期范围（1825天）")
    end_date_str = datetime.now().strftime('%Y%m%d')
    start_date_str = (datetime.now() - timedelta(days=1825)).strftime('%Y%m%d')
    
    print(f"日期范围: {start_date_str} 到 {end_date_str}")
    
    try:
        stock_data = data_provider.get_stock_data(
            stock_code, 
            'daily', 
            start_date_str, 
            end_date_str, 
            save_to_temp_file=False
        )
        
        if not stock_data.empty:
            print(f"✓ 成功获取股票数据，数据量: {len(stock_data)}")
            print(f"  列名: {list(stock_data.columns)}")
            if 'date' in stock_data.columns:
                print(f"  数据范围: {stock_data['date'].min()} 到 {stock_data['date'].max()}")
                print(f"  实际数据天数: {(stock_data['date'].max() - stock_data['date'].min()).days} 天")
            print(f"  数据行数: {len(stock_data)} 行")
        else:
            print("✗ 获取到的数据为空")
            
    except Exception as e:
        print(f"✗ 获取数据失败: {e}")
    
    # 测试2: 检查数据提供者内部逻辑
    print("\n测试2: 检查数据提供者内部逻辑")
    print("检查get_stock_data方法中的日期参数处理...")
    
    # 查看方法源码
    import inspect
    source = inspect.getsource(data_provider.get_stock_data)
    
    # 查找日期参数处理逻辑
    lines = source.split('\n')
    date_logic_found = False
    for i, line in enumerate(lines):
        if 'start_date' in line and 'end_date' in line:
            print(f"找到日期参数处理逻辑 (行 {i+1}): {line.strip()}")
            date_logic_found = True
        if 'timedelta(days=1095)' in line:
            print(f"找到默认日期范围设置 (行 {i+1}): {line.strip()}")
            date_logic_found = True
    
    if not date_logic_found:
        print("未找到明显的日期参数处理逻辑")

def test_direct_akshare_with_dates():
    """直接测试akshare获取数据"""
    print("\n" + "=" * 60)
    print("直接测试akshare获取数据")
    print("=" * 60)
    
    import akshare as ak
    
    stock_code = "600523"
    end_date_str = datetime.now().strftime('%Y%m%d')
    start_date_str = (datetime.now() - timedelta(days=1825)).strftime('%Y%m%d')
    
    print(f"直接测试akshare，日期范围: {start_date_str} 到 {end_date_str}")
    
    try:
        stock_data = ak.stock_zh_a_hist(
            symbol=stock_code, 
            period='daily', 
            start_date=start_date_str, 
            end_date=end_date_str,
            adjust="hfq"
        )
        
        if not stock_data.empty:
            print(f"✓ 直接akshare获取成功，数据量: {len(stock_data)}")
            print(f"  列名: {list(stock_data.columns)}")
            if '日期' in stock_data.columns:
                print(f"  数据范围: {stock_data['日期'].min()} 到 {stock_data['日期'].max()}")
            print(f"  数据行数: {len(stock_data)} 行")
        else:
            print("✗ 直接akshare获取到的数据为空")
            
    except Exception as e:
        print(f"✗ 直接akshare获取失败: {e}")

def test_app_py_date_logic():
    """测试app.py中的日期逻辑"""
    print("\n" + "=" * 60)
    print("测试app.py中的日期逻辑")
    print("=" * 60)
    
    # 读取app.py文件
    app_py_path = os.path.join(os.path.dirname(__file__), 'app.py')
    
    try:
        with open(app_py_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找日期参数设置逻辑
        lines = content.split('\n')
        date_logic_found = False
        
        for i, line in enumerate(lines):
            if 'start_date_str' in line and 'timedelta(days=1825)' in line:
                print(f"找到5年日期范围设置 (行 {i+1}): {line.strip()}")
                date_logic_found = True
            if 'get_stock_data' in line and 'start_date_str' in line:
                print(f"找到数据获取调用 (行 {i+1}): {line.strip()}")
                date_logic_found = True
        
        if not date_logic_found:
            print("未找到明显的日期参数设置逻辑")
            
    except Exception as e:
        print(f"读取app.py失败: {e}")

if __name__ == "__main__":
    test_data_provider_date_params()
    test_direct_akshare_with_dates()
    test_app_py_date_logic()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)