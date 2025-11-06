#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试缓存和日期参数问题
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from akshare_data_provider import AkshareDataProvider
import pandas as pd
from datetime import datetime, timedelta

def test_cache_date_issue():
    """测试缓存和日期参数问题"""
    print("=" * 60)
    print("测试缓存和日期参数问题")
    print("=" * 60)
    
    # 创建数据提供者实例
    data_provider = AkshareDataProvider()
    
    # 测试股票代码
    stock_code = "600523"
    
    # 测试1: 使用5年日期范围
    print("\n测试1: 使用5年日期范围")
    end_date_str = datetime.now().strftime('%Y%m%d')
    start_date_str = (datetime.now() - timedelta(days=1825)).strftime('%Y%m%d')
    
    print(f"传入的日期参数: start_date={start_date_str}, end_date={end_date_str}")
    
    try:
        # 第一次调用
        stock_data = data_provider.get_stock_data(
            stock_code, 
            'daily', 
            start_date_str, 
            end_date_str, 
            save_to_temp_file=False
        )
        
        if not stock_data.empty:
            print(f"✓ 第一次调用成功，数据量: {len(stock_data)}")
            if 'date' in stock_data.columns:
                print(f"  数据范围: {stock_data['date'].min()} 到 {stock_data['date'].max()}")
        else:
            print("✗ 第一次调用获取到的数据为空")
            
    except Exception as e:
        print(f"✗ 第一次调用失败: {e}")
    
    # 测试2: 检查缓存键生成
    print("\n测试2: 检查缓存键生成")
    cache_key = f"stock_{stock_code}_daily_{start_date_str}_{end_date_str}"
    print(f"预期的缓存键: {cache_key}")
    
    # 检查缓存中是否有这个键
    if hasattr(data_provider, 'cache'):
        if cache_key in data_provider.cache:
            print("✓ 缓存键存在于缓存中")
            cached_data = data_provider.cache[cache_key]
            print(f"  缓存数据量: {len(cached_data)}")
            if 'date' in cached_data.columns:
                print(f"  缓存数据范围: {cached_data['date'].min()} 到 {cached_data['date'].max()}")
        else:
            print("✗ 缓存键不存在于缓存中")
            
            # 检查缓存中的所有键
            print("\n缓存中的所有键:")
            for key in list(data_provider.cache.keys())[:10]:  # 只显示前10个
                print(f"  {key}")
    
    # 测试3: 直接检查数据提供者内部逻辑
    print("\n测试3: 检查数据提供者内部逻辑")
    
    # 模拟数据提供者内部的日期处理逻辑
    test_start_date = start_date_str
    test_end_date = end_date_str
    
    print(f"传入的日期参数: start_date={test_start_date}, end_date={test_end_date}")
    
    # 模拟数据提供者内部的默认日期设置逻辑
    if not test_end_date:
        test_end_date = datetime.now().strftime('%Y%m%d')
        print(f"end_date为空，设置为默认值: {test_end_date}")
    
    if not test_start_date:
        test_start_date = (datetime.now() - timedelta(days=1095)).strftime('%Y%m%d')  # 3年
        print(f"start_date为空，设置为默认值: {test_start_date}")
    
    print(f"最终使用的日期参数: start_date={test_start_date}, end_date={test_end_date}")
    
    # 测试4: 检查临时文件读取逻辑
    print("\n测试4: 检查临时文件读取逻辑")
    
    # 检查临时目录
    temp_dir = os.path.join(os.path.expanduser('~'), 'AppData', 'Local', 'Temp', 'kronos')
    print(f"临时目录: {temp_dir}")
    
    if os.path.exists(temp_dir):
        # 查找以股票代码命名的临时文件
        import glob
        pattern = os.path.join(temp_dir, f"stock_{stock_code}_*.csv")
        temp_files = glob.glob(pattern)
        
        if temp_files:
            print(f"找到临时文件: {len(temp_files)} 个")
            for temp_file in temp_files[:3]:  # 只显示前3个
                print(f"  {temp_file}")
                
                # 读取临时文件内容
                try:
                    temp_data = pd.read_csv(temp_file)
                    print(f"    临时文件数据量: {len(temp_data)}")
                    if 'date' in temp_data.columns:
                        temp_data['date'] = pd.to_datetime(temp_data['date'])
                        print(f"    临时文件数据范围: {temp_data['date'].min()} 到 {temp_data['date'].max()}")
                except Exception as e:
                    print(f"    读取临时文件失败: {e}")
        else:
            print("未找到临时文件")
    else:
        print("临时目录不存在")

def test_direct_call_without_cache():
    """测试直接调用akshare，不使用缓存"""
    print("\n" + "=" * 60)
    print("测试直接调用akshare，不使用缓存")
    print("=" * 60)
    
    import akshare as ak
    
    stock_code = "600523"
    end_date_str = datetime.now().strftime('%Y%m%d')
    start_date_str = (datetime.now() - timedelta(days=1825)).strftime('%Y%m%d')
    
    print(f"直接调用akshare，日期范围: {start_date_str} 到 {end_date_str}")
    
    try:
        stock_data = ak.stock_zh_a_hist(
            symbol=stock_code, 
            period='daily', 
            start_date=start_date_str, 
            end_date=end_date_str,
            adjust="hfq"
        )
        
        if not stock_data.empty:
            print(f"✓ 直接akshare调用成功，数据量: {len(stock_data)}")
            if '日期' in stock_data.columns:
                stock_data['日期'] = pd.to_datetime(stock_data['日期'])
                print(f"  数据范围: {stock_data['日期'].min()} 到 {stock_data['日期'].max()}")
                print(f"  实际数据天数: {(stock_data['日期'].max() - stock_data['日期'].min()).days} 天")
        else:
            print("✗ 直接akshare调用获取到的数据为空")
            
    except Exception as e:
        print(f"✗ 直接akshare调用失败: {e}")

if __name__ == "__main__":
    test_cache_date_issue()
    test_direct_call_without_cache()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)