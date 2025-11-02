import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'webui'))

from akshare_data_provider import AkshareDataProvider

def test_search_function():
    """测试数据提供器的搜索功能"""
    print("初始化Akshare数据提供器...")
    provider = AkshareDataProvider()
    
    # 测试不同的搜索关键词
    test_keywords = ["平安银行", "000001", "600519", "贵州茅台"]
    
    for keyword in test_keywords:
        print(f"\n{'='*50}")
        print(f"测试搜索关键词: {keyword}")
        print(f"{'='*50}")
        
        try:
            results = provider.search_stock(keyword)
            print(f"搜索结果数量: {len(results)}")
            
            if results:
                print("前3个搜索结果:")
                for i, stock in enumerate(results[:3]):
                    print(f"  {i+1}. 代码: {stock.get('symbol', 'N/A')}")
                    print(f"     名称: {stock.get('name', 'N/A')}")
                    print(f"     最新价: {stock.get('latest_price', 'N/A')}")
                    print(f"     涨跌幅: {stock.get('change_rate', 'N/A')}")
            else:
                print("未找到匹配的股票")
                
        except Exception as e:
            print(f"搜索失败: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_search_function()