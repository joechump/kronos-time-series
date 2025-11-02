import akshare as ak
from datetime import datetime, timedelta

print("测试akshare历史数据接口...")

try:
    # 获取最近一周的历史数据
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d')
    
    print(f"获取股票600135从{start_date}到{end_date}的历史数据...")
    hist_data = ak.stock_zh_a_hist(symbol="600135", period='daily', start_date=start_date, end_date=end_date)
    
    if not hist_data.empty:
        print("成功获取历史数据:")
        print(f"数据形状: {hist_data.shape}")
        print("列名:", list(hist_data.columns))
        print("\n最新一行数据:")
        latest_row = hist_data.iloc[-1]
        print(latest_row)
        
        # 检查关键列是否存在
        key_columns = ['名称', '收盘', '涨跌幅', '涨跌额', '成交量', '成交额']
        for col in key_columns:
            if col in hist_data.columns:
                print(f"  {col}: {latest_row[col]}")
            else:
                print(f"  {col}: 不存在")
    else:
        print("历史数据为空")
        
except Exception as e:
    print(f"获取历史股票数据失败: {e}")
    import traceback
    traceback.print_exc()