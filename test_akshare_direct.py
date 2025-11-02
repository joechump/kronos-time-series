import akshare as ak
import pandas as pd

print("测试akshare库是否能正常工作...")

try:
    # 测试实时数据接口
    print("尝试获取实时股票数据...")
    stock_list = ak.stock_zh_a_spot_em()
    print(f"成功获取到 {len(stock_list)} 只股票的数据")
    
    # 查找特定股票
    keyword = "600135"
    results = stock_list[stock_list['代码'].str.contains(keyword)]
    if not results.empty:
        print(f"找到股票 {keyword}:")
        row = results.iloc[0]
        print(f"  名称: {row['名称']}")
        print(f"  最新价: {row['最新价']}")
        print(f"  涨跌幅: {row['涨跌幅']}")
        print(f"  涨跌额: {row['涨跌额']}")
        print(f"  成交量: {row['成交量']}")
        print(f"  成交额: {row['成交额']}")
    else:
        print(f"未找到股票 {keyword}")
        
except Exception as e:
    print(f"获取实时股票数据失败: {e}")
    import traceback
    traceback.print_exc()

try:
    # 测试历史数据接口
    print("\n尝试获取历史股票数据...")
    hist_data = ak.stock_zh_a_hist(symbol="600135", period='daily', start_date='20241027', end_date='20241028')
    if not hist_data.empty:
        print("成功获取历史数据:")
        print(hist_data.head())
    else:
        print("历史数据为空")
        
except Exception as e:
    print(f"获取历史股票数据失败: {e}")
    import traceback
    traceback.print_exc()