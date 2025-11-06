#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试akshare基本功能
"""

import sys
import os
import time
import akshare as ak
import pandas as pd

def test_akshare_basic():
    """测试akshare基本功能"""
    print("=" * 60)
    print("测试akshare基本功能")
    print("=" * 60)
    
    try:
        # 测试1: 获取股票实时数据
        print("1. 测试获取股票实时数据...")
        start_time = time.time()
        try:
            stock_spot = ak.stock_zh_a_spot_em()
            end_time = time.time()
            print(f"✓ 实时数据获取成功，耗时: {end_time - start_time:.2f}秒")
            print(f"  数据量: {len(stock_spot)} 只股票")
            if len(stock_spot) > 0:
                print(f"  示例股票: {stock_spot.iloc[0]['代码']} - {stock_spot.iloc[0]['名称']}")
        except Exception as e:
            print(f"✗ 实时数据获取失败: {e}")
        
        # 测试2: 获取茅台历史数据（简化版）
        print("\n2. 测试获取茅台历史数据...")
        start_time = time.time()
        try:
            # 只获取最近10天的数据，避免长时间等待
            stock_hist = ak.stock_zh_a_hist(symbol="600519", period="daily", start_date="20241025", end_date="20241104")
            end_time = time.time()
            print(f"✓ 历史数据获取成功，耗时: {end_time - start_time:.2f}秒")
            print(f"  数据量: {len(stock_hist)} 条记录")
            if len(stock_hist) > 0:
                print(f"  最新数据: {stock_hist.iloc[-1]['日期']} - 收盘价: {stock_hist.iloc[-1]['收盘']}")
        except Exception as e:
            print(f"✗ 历史数据获取失败: {e}")
        
        # 测试3: 测试股票基本信息
        print("\n3. 测试股票基本信息...")
        start_time = time.time()
        try:
            stock_info = ak.stock_individual_info_em(symbol="600519")
            end_time = time.time()
            print(f"✓ 股票信息获取成功，耗时: {end_time - start_time:.2f}秒")
            print(f"  信息条目: {len(stock_info)} 条")
        except Exception as e:
            print(f"✗ 股票信息获取失败: {e}")
        
        print("\n" + "=" * 60)
        print("akshare基本功能测试完成")
        print("=" * 60)
        
    except Exception as e:
        print(f"✗ akshare测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_akshare_basic()