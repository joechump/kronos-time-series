import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'webui'))

# 测试直接导入AkshareDataProvider
try:
    from webui.akshare_data_provider import AkshareDataProvider
    print("成功导入AkshareDataProvider")
    
    # 测试初始化
    provider = AkshareDataProvider()
    print("成功初始化AkshareDataProvider实例")
    
    # 测试搜索功能
    print("测试搜索功能...")
    results = provider.search_stock("平安银行")
    print(f"搜索'平安银行'返回结果数量: {len(results)}")
    
    # 测试代码搜索
    print("测试代码搜索...")
    results2 = provider.search_stock("000001")
    print(f"搜索'000001'返回结果数量: {len(results2)}")
    
except Exception as e:
    print(f"导入或初始化AkshareDataProvider失败: {e}")
    import traceback
    traceback.print_exc()

# 测试app.py中的data_provider初始化
print("\n测试app.py中的data_provider初始化...")
try:
    # 模拟app.py中的初始化过程
    try:
        from webui.akshare_data_provider import AkshareDataProvider
        data_provider = AkshareDataProvider()
        print("app.py中的data_provider初始化成功")
        
        # 测试搜索功能
        print("测试搜索功能...")
        results = data_provider.search_stock("平安银行")
        print(f"搜索'平安银行'返回结果数量: {len(results)}")
        
    except ImportError as e:
        print(f"app.py中的data_provider初始化失败: {e}")
        data_provider = None
        
except Exception as e:
    print(f"app.py中的data_provider测试失败: {e}")
    import traceback
    traceback.print_exc()