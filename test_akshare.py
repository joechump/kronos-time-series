import akshare as ak
import pandas as pd

# 测试实时数据接口
try:
    print("正在测试实时数据接口...")
    stock_list = ak.stock_zh_a_spot_em()
    print(f"实时数据接口返回数据量: {len(stock_list)}")
    
    # 查找乐凯胶片(600135)
    result = stock_list[stock_list['代码'] == '600135']
    if not result.empty:
        print("找到乐凯胶片的数据:")
        print(result[['代码', '名称', '最新价', '涨跌幅', '涨跌额', '成交量', '成交额']])
    else:
        print("未找到乐凯胶片的数据")
        
    # 显示列名
    print("\n所有列名:")
    print(list(stock_list.columns))
    
except Exception as e:
    print(f"实时数据接口测试失败: {e}")
    import traceback
    traceback.print_exc()