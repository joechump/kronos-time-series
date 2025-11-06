#!/usr/bin/env python3
"""
测试修复后的股票搜索功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from akshare_data_provider import AkshareDataProvider

def test_final_search():
    """测试修复后的股票搜索功能"""
    print("=" * 50)
    print("测试修复后的股票搜索功能")
    print("=" * 50)
    
    provider = AkshareDataProvider()
    
    # 测试1: 搜索股票代码 '600519'
    print("\n1. 测试股票代码 '600519':")
    try:
        results = provider.search_stock('600519')
        print(f"搜索结果数量: {len(results)}")
        if results:
            for stock in results:
                print(f"  股票代码: {stock['symbol']}")
                print(f"  股票名称: {stock['name']}")
                print(f"  当前价格: {stock['latest_price']}")
                print(f"  涨跌幅: {stock['change_rate']}")
                print(f"  涨跌额: {stock['change_amount']}")
        else:
            print("  未找到结果")
    except Exception as e:
        print(f"  搜索失败: {e}")
    
    # 测试2: 搜索股票名称 '茅台'
    print("\n2. 测试股票名称 '茅台':")
    try:
        results = provider.search_stock('茅台')
        print(f"搜索结果数量: {len(results)}")
        if results:
            for stock in results:
                print(f"  股票代码: {stock['symbol']}")
                print(f"  股票名称: {stock['name']}")
                print(f"  当前价格: {stock['latest_price']}")
                print(f"  涨跌幅: {stock['change_rate']}")
                print(f"  涨跌额: {stock['change_amount']}")
        else:
            print("  未找到结果")
    except Exception as e:
        print(f"  搜索失败: {e}")
    
    # 测试3: 搜索股票名称 '贵州茅台'
    print("\n3. 测试股票名称 '贵州茅台':")
    try:
        results = provider.search_stock('贵州茅台')
        print(f"搜索结果数量: {len(results)}")
        if results:
            for stock in results:
                print(f"  股票代码: {stock['symbol']}")
                print(f"  股票名称: {stock['name']}")
                print(f"  当前价格: {stock['latest_price']}")
                print(f"  涨跌幅: {stock['change_rate']}")
                print(f"  涨跌额: {stock['change_amount']}")
        else:
            print("  未找到结果")
    except Exception as e:
        print(f"  搜索失败: {e}")
    
    # 测试4: 搜索不存在的股票
    print("\n4. 测试不存在的股票 '不存在的股票':")
    try:
        results = provider.search_stock('不存在的股票')
        print(f"搜索结果数量: {len(results)}")
        if results:
            for stock in results:
                print(f"  股票代码: {stock['symbol']}")
                print(f"  股票名称: {stock['name']}")
                print(f"  当前价格: {stock['latest_price']}")
                print(f"  涨跌幅: {stock['change_rate']}")
                print(f"  涨跌额: {stock['change_amount']}")
        else:
            print("  未找到结果（正确行为）")
    except Exception as e:
        print(f"  搜索失败: {e}")
    
    print("\n测试完成!")

if __name__ == "__main__":
    test_final_search()