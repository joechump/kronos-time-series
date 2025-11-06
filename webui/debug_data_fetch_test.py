"""
测试akshare数据获取功能，诊断数据量不足问题
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import datetime
from akshare_data_provider import AkshareDataProvider

def test_akshare_data_fetch():
    """测试akshare数据获取"""
    print("=== 测试akshare数据获取功能 ===")
    
    # 创建数据提供者
    provider = AkshareDataProvider()
    
    # 测试股票代码
    test_stocks = ['600519', '000001', '000858']
    
    for stock_code in test_stocks:
        print(f"\n--- 测试股票 {stock_code} ---")
        
        # 设置日期范围（3年）
        end_date = datetime.datetime.now().strftime('%Y%m%d')
        start_date = (datetime.datetime.now() - datetime.timedelta(days=1095)).strftime('%Y%m%d')
        
        print(f"日期范围: {start_date} 到 {end_date}")
        
        # 获取数据
        try:
            stock_data = provider.get_stock_data(stock_code, 'daily', start_date, end_date)
            
            if stock_data is None or stock_data.empty:
                print(f"股票 {stock_code}: 未获取到数据")
                continue
                
            print(f"股票 {stock_code}: 获取到 {len(stock_data)} 行数据")
            print(f"数据列: {list(stock_data.columns)}")
            
            if len(stock_data) > 0:
                print(f"日期范围: {stock_data['date'].min()} 到 {stock_data['date'].max()}")
                print(f"前5行数据:")
                print(stock_data.head())
                
                # 检查数据质量
                print(f"数据质量检查:")
                print(f"  空值统计:")
                print(stock_data.isnull().sum())
                
        except Exception as e:
            print(f"股票 {stock_code}: 获取数据失败 - {e}")

def test_specific_date_range():
    """测试特定日期范围的数据获取"""
    print("\n=== 测试特定日期范围 ===")
    
    provider = AkshareDataProvider()
    stock_code = '600519'
    
    # 测试不同的日期范围
    date_ranges = [
        ('20240101', '20241231'),  # 2024年全年
        ('20230101', '20231231'),  # 2023年全年
        ('20220101', '20221231'),  # 2022年全年
        ('20210101', '20211231'),  # 2021年全年
    ]
    
    for start_date, end_date in date_ranges:
        print(f"\n日期范围: {start_date} 到 {end_date}")
        
        try:
            stock_data = provider.get_stock_data(stock_code, 'daily', start_date, end_date)
            
            if stock_data is None or stock_data.empty:
                print(f"  未获取到数据")
            else:
                print(f"  获取到 {len(stock_data)} 行数据")
                if len(stock_data) > 0:
                    print(f"  实际日期范围: {stock_data['date'].min()} 到 {stock_data['date'].max()}")
                    
        except Exception as e:
            print(f"  获取数据失败 - {e}")

if __name__ == "__main__":
    test_akshare_data_fetch()
    test_specific_date_range()