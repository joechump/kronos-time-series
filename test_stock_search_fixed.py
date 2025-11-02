import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'webui'))

from akshare_data_provider import AkshareDataProvider

def test_stock_search():
    """测试修复后的股票搜索功能"""
    provider = AkshareDataProvider()
    
    # 测试搜索贵州茅台（600519）
    print("测试搜索贵州茅台（600519）...")
    try:
        results = provider.search_stock("600519")
        print(f"搜索结果: {results}")
        
        if results:
            first_result = results[0]
            print(f"股票代码: {first_result.get('symbol', 'N/A')}")
            print(f"股票名称: {first_result.get('name', 'N/A')}")
            print(f"最新价格: {first_result.get('latest_price', 'N/A')}")
            print(f"涨跌幅: {first_result.get('change_rate', 'N/A')}")
            print("测试成功！")
        else:
            print("未找到搜索结果")
    except Exception as e:
        print(f"搜索失败: {e}")

if __name__ == "__main__":
    test_stock_search()