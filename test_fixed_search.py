import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'webui'))

from akshare_data_provider import AkshareDataProvider

def test_fixed_search():
    """
    测试修复后的股票搜索功能
    """
    # 初始化数据提供器
    data_provider = AkshareDataProvider()
    
    # 测试搜索功能
    test_keywords = ["平安银行", "000001", "600519", "贵州茅台"]
    
    for keyword in test_keywords:
        print(f"\n测试搜索关键词: {keyword}")
        try:
            results = data_provider.search_stock(keyword)
            print(f"搜索结果数量: {len(results)}")
            for result in results:
                print(f"  - {result['symbol']}: {result['name']}")
                print(f"    最新价: {result['latest_price']}, 涨跌幅: {result['change_rate']}%")
        except Exception as e:
            print(f"搜索失败: {e}")

if __name__ == "__main__":
    print("测试修复后的股票搜索功能")
    test_fixed_search()