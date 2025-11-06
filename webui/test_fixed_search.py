#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修复后的股票搜索功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from akshare_data_provider import AkshareDataProvider

def test_search_functionality():
    """测试修复后的股票搜索功能"""
    print("=" * 50)
    print("测试修复后的股票搜索功能")
    print("=" * 50)
    
    # 创建数据提供者实例
    provider = AkshareDataProvider()
    
    # 测试1: 搜索股票代码600519
    print("\n1. 测试搜索股票代码 '600519':")
    results = provider.search_stock("600519")
    print(f"搜索结果数量: {len(results)}")
    if results:
        for stock in results:
            print(f"  代码: {stock['symbol']}, 名称: {stock['name']}, 价格: {stock['latest_price']}")
    
    # 测试2: 搜索股票名称"茅台"
    print("\n2. 测试搜索股票名称 '茅台':")
    results = provider.search_stock("茅台")
    print(f"搜索结果数量: {len(results)}")
    if results:
        for stock in results:
            print(f"  代码: {stock['symbol']}, 名称: {stock['name']}, 价格: {stock['latest_price']}")
    
    # 测试3: 搜索股票名称"贵州茅台"
    print("\n3. 测试搜索股票名称 '贵州茅台':")
    results = provider.search_stock("贵州茅台")
    print(f"搜索结果数量: {len(results)}")
    if results:
        for stock in results:
            print(f"  代码: {stock['symbol']}, 名称: {stock['name']}, 价格: {stock['latest_price']}")
    
    # 测试4: 搜索不存在的股票代码
    print("\n4. 测试搜索不存在的股票代码 '999999':")
    results = provider.search_stock("999999")
    print(f"搜索结果数量: {len(results)}")
    if results:
        for stock in results:
            print(f"  代码: {stock['symbol']}, 名称: {stock['name']}, 价格: {stock['latest_price']}")
    
    # 测试5: 搜索不存在的股票名称
    print("\n5. 测试搜索不存在的股票名称 '不存在的股票':")
    results = provider.search_stock("不存在的股票")
    print(f"搜索结果数量: {len(results)}")
    if results:
        for stock in results:
            print(f"  代码: {stock['symbol']}, 名称: {stock['name']}, 价格: {stock['latest_price']}")

if __name__ == "__main__":
    test_search_functionality()
    print("\n测试完成!")