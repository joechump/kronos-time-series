#!/usr/bin/env python3
"""
测试akshare是否能正确获取600519股票数据
"""

import akshare as ak
import pandas as pd
import sys
from datetime import datetime, timedelta

def test_akshare_600519():
    """测试akshare获取600519股票数据"""
    print("=== 测试akshare获取600519股票数据 ===")
    
    # 测试不同的股票代码格式
    test_codes = ["600519", "000001", "000858"]
    
    for symbol in test_codes:
        print(f"\n--- 测试股票代码: {symbol} ---")
        
        # 设置日期范围
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')  # 最近30天
        
        try:
            # 尝试获取股票数据
            print(f"尝试获取 {symbol} 的股票数据...")
            stock_data = ak.stock_zh_a_hist(
                symbol=symbol, 
                period='daily', 
                start_date=start_date, 
                end_date=end_date,
                adjust="hfq"
            )
            
            if not stock_data.empty:
                print(f"✅ 成功获取 {symbol} 的数据，共 {len(stock_data)} 条记录")
                print(f"数据列: {list(stock_data.columns)}")
                print(f"日期范围: {stock_data.iloc[0]['日期']} 到 {stock_data.iloc[-1]['日期']}")
                print(f"最新收盘价: {stock_data.iloc[-1]['收盘']}")
            else:
                print(f"❌ 获取到空数据: {symbol}")
                
        except Exception as e:
            print(f"❌ 获取 {symbol} 数据失败: {e}")
            
            # 尝试其他可能的接口
            try:
                print(f"尝试备用接口获取 {symbol}...")
                # 尝试实时数据接口
                spot_data = ak.stock_zh_a_spot_em()
                if not spot_data.empty:
                    # 查找特定股票
                    target_stock = spot_data[spot_data['代码'] == symbol]
                    if not target_stock.empty:
                        print(f"✅ 通过实时接口找到 {symbol}: {target_stock.iloc[0]['名称']}")
                        print(f"最新价: {target_stock.iloc[0]['最新价']}")
                    else:
                        print(f"❌ 实时接口中未找到 {symbol}")
                else:
                    print("❌ 实时接口返回空数据")
                    
            except Exception as e2:
                print(f"❌ 备用接口也失败: {e2}")

def test_akshare_availability():
    """测试akshare整体可用性"""
    print("\n=== 测试akshare整体可用性 ===")
    
    try:
        # 测试获取股票列表
        print("测试获取股票列表...")
        stock_list = ak.stock_zh_a_spot_em()
        if not stock_list.empty:
            print(f"✅ 成功获取股票列表，共 {len(stock_list)} 只股票")
            print(f"前5只股票: {list(stock_list.head()[['代码', '名称']].values)}")
            
            # 检查是否包含600519
            has_600519 = '600519' in stock_list['代码'].values
            print(f"股票列表中是否包含600519: {has_600519}")
            
            if has_600519:
                stock_600519 = stock_list[stock_list['代码'] == '600519'].iloc[0]
                print(f"600519信息: {stock_600519['名称']}, 最新价: {stock_600519['最新价']}")
        else:
            print("❌ 股票列表为空")
            
    except Exception as e:
        print(f"❌ 获取股票列表失败: {e}")

if __name__ == "__main__":
    print("akshare版本:", ak.__version__)
    print("pandas版本:", pd.__version__)
    
    test_akshare_availability()
    test_akshare_600519()
    
    print("\n=== 测试完成 ===")