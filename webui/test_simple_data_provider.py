#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版数据提供者测试脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from akshare_data_provider import AkshareDataProvider
import pandas as pd
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_basic_functionality():
    """测试基本功能"""
    print("=" * 60)
    print("测试数据提供者基本功能")
    print("=" * 60)
    
    try:
        # 创建数据提供者实例
        print("1. 创建数据提供者实例...")
        provider = AkshareDataProvider()
        print("✓ 数据提供者实例创建成功")
        
        # 测试获取股票数据
        print("\n2. 测试获取股票数据...")
        try:
            stock_data = provider.get_stock_data("600519", period="daily", start_date="20230101", end_date="20231231")
            if not stock_data.empty:
                print(f"✓ 成功获取股票数据，数据量: {len(stock_data)}")
                print(f"  数据列: {list(stock_data.columns)}")
                print(f"  日期范围: {stock_data['date'].min()} 到 {stock_data['date'].max()}")
            else:
                print("⚠ 获取到空数据")
        except Exception as e:
            print(f"✗ 获取股票数据失败: {e}")
        
        # 测试搜索功能（简化版）
        print("\n3. 测试搜索功能（简化版）...")
        try:
            # 直接测试搜索茅台
            import akshare as ak
            stock_list = ak.stock_zh_a_spot_em()
            print(f"✓ 股票列表获取成功，共 {len(stock_list)} 只股票")
            
            # 搜索茅台
            results = stock_list[stock_list['名称'].str.contains('茅台', case=False)]
            if not results.empty:
                print(f"✓ 搜索成功，找到 {len(results)} 只相关股票")
                for _, row in results.head(3).iterrows():
                    print(f"  代码: {row['代码']}, 名称: {row['名称']}, 最新价: {row['最新价']}")
            else:
                print("⚠ 未找到相关股票")
        except Exception as e:
            print(f"✗ 搜索功能测试失败: {e}")
        
        print("\n" + "=" * 60)
        print("基本功能测试完成")
        print("=" * 60)
        
    except Exception as e:
        print(f"✗ 数据提供者测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_basic_functionality()