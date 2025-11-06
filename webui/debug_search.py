#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试股票搜索功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import akshare as ak
import pandas as pd

def debug_search():
    """调试股票搜索功能"""
    print("=" * 50)
    print("调试股票搜索功能")
    print("=" * 50)
    
    # 测试股票基本信息接口
    print("\n1. 测试股票基本信息接口 '600519':")
    try:
        stock_info = ak.stock_individual_info_em(symbol="600519")
        print("基本信息接口成功:")
        print(stock_info)
        
        # 提取股票名称
        name_row = stock_info[stock_info['item'] == '股票简称']
        if not name_row.empty:
            stock_name = name_row.iloc[0]['value']
            print(f"股票名称: {stock_name}")
        else:
            print("未找到股票简称")
            
    except Exception as e:
        print(f"基本信息接口失败: {e}")
    
    # 测试历史数据接口
    print("\n2. 测试历史数据接口 '600519':")
    try:
        from datetime import datetime, timedelta
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d')
        
        hist_data = ak.stock_zh_a_hist(symbol="600519", period='daily', 
                                      start_date=start_date, end_date=end_date)
        print(f"历史数据接口成功，数据行数: {len(hist_data)}")
        if not hist_data.empty:
            latest_row = hist_data.iloc[-1]
            print(f"最新价格: {latest_row['收盘']}")
            print(f"涨跌幅: {latest_row['涨跌幅']}")
            print(f"涨跌额: {latest_row['涨跌额']}")
            
    except Exception as e:
        print(f"历史数据接口失败: {e}")
    
    # 测试已知股票列表
    print("\n3. 测试已知股票列表匹配:")
    known_stocks = {
        '茅台': '600519',
        '贵州茅台': '600519', 
        '平安': '000001',
        '中国平安': '000001',
        '招商银行': '600036',
        '万科': '000002',
        '万科A': '000002',
        '五粮液': '000858',
        '格力电器': '000651',
        '美的集团': '000333'
    }
    
    keyword = "茅台"
    matched_symbol = None
    for name, symbol in known_stocks.items():
        if keyword in name:
            matched_symbol = symbol
            print(f"关键词 '{keyword}' 匹配到股票: {name} -> {symbol}")
            break
    
    if matched_symbol:
        print(f"匹配的股票代码: {matched_symbol}")
    else:
        print(f"未找到匹配的股票")

if __name__ == "__main__":
    debug_search()
    print("\n调试完成!")