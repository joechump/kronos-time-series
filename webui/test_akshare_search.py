#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试akshare股票搜索功能
"""

import akshare as ak
import pandas as pd

def test_akshare_search():
    """测试akshare的股票搜索功能"""
    print("测试akshare股票搜索功能...")
    
    # 测试实时数据接口
    try:
        print("\n1. 测试实时数据接口...")
        stock_list = ak.stock_zh_a_spot_em()
        print(f"实时数据接口返回 {len(stock_list)} 只股票")
        
        # 搜索贵州茅台
        print("\n2. 搜索贵州茅台(600519)...")
        
        # 按代码搜索
        results_by_code = stock_list[stock_list['代码'].str.contains('600519')]
        print(f"按代码搜索结果数量: {len(results_by_code)}")
        if not results_by_code.empty:
            print("按代码搜索结果:")
            for _, row in results_by_code.iterrows():
                print(f"  代码: {row['代码']}, 名称: {row['名称']}")
        
        # 按名称搜索
        results_by_name = stock_list[stock_list['名称'].str.contains('茅台', case=False)]
        print(f"按名称搜索结果数量: {len(results_by_name)}")
        if not results_by_name.empty:
            print("按名称搜索结果:")
            for _, row in results_by_name.iterrows():
                print(f"  代码: {row['代码']}, 名称: {row['名称']}")
        
        # 查看前几行数据了解格式
        print("\n3. 查看股票列表前5行:")
        print(stock_list.head())
        
        # 查看列名
        print("\n4. 列名:")
        print(stock_list.columns.tolist())
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")

def test_stock_info():
    """测试股票基本信息接口"""
    print("\n5. 测试股票基本信息接口...")
    
    try:
        # 测试贵州茅台的基本信息
        stock_info = ak.stock_individual_info_em(symbol="600519")
        print("贵州茅台基本信息:")
        print(stock_info)
        
    except Exception as e:
        print(f"基本信息接口错误: {e}")

def test_historical_data():
    """测试历史数据接口"""
    print("\n6. 测试历史数据接口...")
    
    try:
        # 测试贵州茅台的历史数据
        hist_data = ak.stock_zh_a_hist(symbol="600519", period='daily', 
                                      start_date="20240101", end_date="20241104")
        print(f"历史数据行数: {len(hist_data)}")
        if not hist_data.empty:
            print("前5行历史数据:")
            print(hist_data.head())
            
    except Exception as e:
        print(f"历史数据接口错误: {e}")

if __name__ == "__main__":
    print("=" * 50)
    print("akshare股票搜索功能测试")
    print("=" * 50)
    
    test_akshare_search()
    test_stock_info()
    test_historical_data()
    
    print("\n测试完成!")