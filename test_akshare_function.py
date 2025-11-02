import akshare as ak
import pandas as pd
import traceback

def test_akshare_functions():
    """测试akshare库的各种功能"""
    print("开始测试akshare库功能...")
    
    # 测试1: 股票实时行情接口
    print("\n=== 测试1: 股票实时行情接口 ===")
    try:
        print("获取A股实时行情...")
        stock_zh_a_spot_em_df = ak.stock_zh_a_spot_em()
        print(f"成功获取到 {len(stock_zh_a_spot_em_df)} 只股票的实时行情")
        print("列名:", stock_zh_a_spot_em_df.columns.tolist())
        
        # 查找茅台股票
        maotai_data = stock_zh_a_spot_em_df[stock_zh_a_spot_em_df['代码'] == '600519']
        if not maotai_data.empty:
            print("找到茅台股票:")
            print(maotai_data[['代码', '名称']])
        else:
            print("未在实时行情中找到茅台股票")
    except Exception as e:
        print(f"股票实时行情接口失败: {e}")
        traceback.print_exc()
    
    # 测试2: 股票基本信息接口
    print("\n=== 测试2: 股票基本信息接口 ===")
    try:
        print("获取茅台股票基本信息...")
        stock_individual_info_em_df = ak.stock_individual_info_em(symbol="600519")
        print("茅台股票基本信息:")
        print(stock_individual_info_em_df)
        
        if not stock_individual_info_em_df.empty:
            # 查找股票简称
            name_row = stock_individual_info_em_df[stock_individual_info_em_df['item'] == '股票简称']
            if not name_row.empty:
                stock_name = name_row.iloc[0]['value']
                print(f"找到股票简称: {stock_name}")
            else:
                print("未找到股票简称")
        else:
            print("股票基本信息为空")
    except Exception as e:
        print(f"股票基本信息接口失败: {e}")
        traceback.print_exc()
    
    # 测试3: 股票历史数据接口
    print("\n=== 测试3: 股票历史数据接口 ===")
    try:
        print("获取茅台股票历史数据...")
        from datetime import datetime, timedelta
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d')
        
        stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol="600519", period="daily", start_date=start_date, end_date=end_date)
        print(f"成功获取到 {len(stock_zh_a_hist_df)} 天的历史数据")
        print("列名:", stock_zh_a_hist_df.columns.tolist())
        
        if not stock_zh_a_hist_df.empty:
            latest_row = stock_zh_a_hist_df.iloc[-1]
            print("最新一天的数据:")
            print(latest_row)
    except Exception as e:
        print(f"股票历史数据接口失败: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_akshare_functions()