import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'webui'))

from akshare_data_provider import AkshareDataProvider
import pandas as pd

def check_stock_data(symbol):
    """检查股票数据量"""
    provider = AkshareDataProvider()
    data = provider.get_stock_data(symbol, 'daily')
    
    if data is not None and not data.empty:
        print(f"股票 {symbol} 数据量: {len(data)}")
        print(f"日期范围: {data['date'].min()} 到 {data['date'].max()}")
        
        # 检查是否满足预测要求(400 + 120 = 520个数据点)
        required_points = 400 + 120  # lookback + pred_len
        if len(data) >= required_points:
            print(f"✅ 数据量满足预测要求 ({required_points} 个数据点)")
        else:
            print(f"❌ 数据量不足，需要 {required_points} 个数据点，当前只有 {len(data)} 个")
    else:
        print(f"❌ 无法获取股票 {symbol} 的数据")

if __name__ == "__main__":
    # 检查茅台股票(600519)的数据量
    check_stock_data('600519')