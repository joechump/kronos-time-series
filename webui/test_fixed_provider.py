#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修复后的数据提供者
验证600519股票数据获取功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from akshare_data_provider import AkshareDataProvider
import pandas as pd

def test_fixed_provider():
    """测试修复后的数据提供者"""
    print("=== 测试修复后的数据提供者 ===")
    
    # 创建数据提供者实例
    provider = AkshareDataProvider()
    
    # 测试600519股票数据获取
    print("\n1. 测试600519股票数据获取...")
    try:
        data_600519 = provider.get_stock_data('600519', 'daily', '20240101', '20241026')
        print(f"600519数据获取结果:")
        print(f"  数据形状: {data_600519.shape}")
        print(f"  数据列: {list(data_600519.columns)}")
        if not data_600519.empty:
            print(f"  数据量: {len(data_600519)}")
            print(f"  日期范围: {data_600519['date'].min()} 到 {data_600519['date'].max()}")
            print(f"  价格范围: {data_600519['close'].min():.2f} - {data_600519['close'].max():.2f}")
            print("  前5行数据:")
            print(data_600519.head())
        else:
            print("  ❌ 未获取到数据")
    except Exception as e:
        print(f"  ❌ 获取600519数据失败: {e}")
    
    # 测试000001股票数据获取（对比测试）
    print("\n2. 测试000001股票数据获取（对比测试）...")
    try:
        data_000001 = provider.get_stock_data('000001', 'daily', '20240101', '20241026')
        print(f"000001数据获取结果:")
        print(f"  数据形状: {data_000001.shape}")
        if not data_000001.empty:
            print(f"  数据量: {len(data_000001)}")
            print(f"  日期范围: {data_000001['date'].min()} 到 {data_000001['date'].max()}")
            print(f"  价格范围: {data_000001['close'].min():.2f} - {data_000001['close'].max():.2f}")
        else:
            print("  ❌ 未获取到数据")
    except Exception as e:
        print(f"  ❌ 获取000001数据失败: {e}")
    
    # 测试模拟数据功能
    print("\n3. 测试模拟数据功能...")
    try:
        # 测试一个不存在的股票代码，应该触发模拟数据
        data_simulated = provider.get_stock_data('999999', 'daily', '20240101', '20241026')
        print(f"模拟数据获取结果:")
        print(f"  数据形状: {data_simulated.shape}")
        if not data_simulated.empty:
            print(f"  数据量: {len(data_simulated)}")
            print(f"  日期范围: {data_simulated['date'].min()} 到 {data_simulated['date'].max()}")
            print(f"  价格范围: {data_simulated['close'].min():.2f} - {data_simulated['close'].max():.2f}")
            print("  前3行数据:")
            print(data_simulated.head(3))
        else:
            print("  ❌ 模拟数据创建失败")
    except Exception as e:
        print(f"  ❌ 模拟数据测试失败: {e}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_fixed_provider()