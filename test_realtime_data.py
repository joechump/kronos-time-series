import akshare as ak
import pandas as pd

print("测试akshare实时数据接口...")

try:
    print("正在获取实时股票数据...")
    stock_list = ak.stock_zh_a_spot_em()
    print(f"成功获取到 {len(stock_list)} 只股票的数据")
    
    # 查找乐凯胶片(600135)
    result = stock_list[stock_list['代码'] == '600135']
    if not result.empty:
        print("找到乐凯胶片的数据:")
        print(result[['代码', '名称', '最新价', '涨跌幅', '涨跌额', '成交量', '成交额']])
        
        # 检查关键字段是否有有效值
        row = result.iloc[0]
        print("\n关键字段检查:")
        print(f"  最新价: {row['最新价']} (类型: {type(row['最新价'])})")
        print(f"  涨跌幅: {row['涨跌幅']} (类型: {type(row['涨跌幅'])})")
        print(f"  涨跌额: {row['涨跌额']} (类型: {type(row['涨跌额'])})")
        print(f"  成交量: {row['成交量']} (类型: {type(row['成交量'])})")
        print(f"  成交额: {row['成交额']} (类型: {type(row['成交额'])})")
        
        # 检查是否有NaN值
        print(f"  最新价是否为NaN: {pd.isna(row['最新价'])}")
        print(f"  涨跌幅是否为NaN: {pd.isna(row['涨跌幅'])}")
        print(f"  涨跌额是否为NaN: {pd.isna(row['涨跌额'])}")
        print(f"  成交量是否为NaN: {pd.isna(row['成交量'])}")
        print(f"  成交额是否为NaN: {pd.isna(row['成交额'])}")
    else:
        print("未找到乐凯胶片的数据")
        
    # 显示列名
    print("\n所有列名:")
    print(list(stock_list.columns))
    
except Exception as e:
    print(f"实时数据接口测试失败: {e}")
    import traceback
    traceback.print_exc()