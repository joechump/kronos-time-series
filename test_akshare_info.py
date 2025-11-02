import akshare as ak
import pandas as pd

# 测试茅台股票代码
keyword = "600519"

print(f"测试股票代码: {keyword}")

# 方法1: 尝试使用股票基本信息接口
try:
    print("尝试使用股票基本信息接口...")
    stock_info = ak.stock_individual_info_em(symbol=keyword)
    print("股票基本信息:")
    print(stock_info)
    
    if not stock_info.empty:
        # 查找股票简称
        name_row = stock_info[stock_info['item'] == '股票简称']
        if not name_row.empty:
            stock_name = name_row.iloc[0]['value']
            print(f"找到股票简称: {stock_name}")
        else:
            print("未找到股票简称")
            
        # 查找所有包含"名称"或"简称"的字段
        name_rows = stock_info[stock_info['item'].str.contains('名称|简称', case=False, na=False)]
        if not name_rows.empty:
            print("找到包含'名称'或'简称'的字段:")
            print(name_rows)
        else:
            print("未找到包含'名称'或'简称'的字段")
    else:
        print("股票基本信息为空")
        
except Exception as e:
    print(f"股票基本信息接口失败: {e}")
    import traceback
    traceback.print_exc()

# 方法2: 尝试使用实时行情接口
try:
    print("\n尝试使用实时行情接口...")
    stock_zh_a_spot_em_df = ak.stock_zh_a_spot_em(symbol=keyword)
    print("实时行情数据:")
    print(stock_zh_a_spot_em_df)
    
    if not stock_zh_a_spot_em_df.empty:
        # 获取股票名称
        if '名称' in stock_zh_a_spot_em_df.columns:
            stock_name = stock_zh_a_spot_em_df['名称'].iloc[0]
            print(f"从实时行情获取股票名称: {stock_name}")
        else:
            print("实时行情数据中没有'名称'列")
    else:
        print("实时行情数据为空")
        
except Exception as e:
    print(f"实时行情接口失败: {e}")
    import traceback
    traceback.print_exc()