#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据提供者状态脚本
"""

import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_data_provider_import():
    """测试数据提供者导入"""
    print("=" * 60)
    print("测试数据提供者导入状态")
    print("=" * 60)
    
    try:
        from akshare_data_provider import AkshareDataProvider
        print("✓ 成功导入AkshareDataProvider")
        
        # 测试初始化
        try:
            provider = AkshareDataProvider()
            print("✓ 成功初始化AkshareDataProvider实例")
            
            # 测试搜索功能
            print("\n测试搜索功能...")
            results = provider.search_stock("600519")
            print(f"✓ 搜索'600519'返回结果数量: {len(results)}")
            
            if results:
                for stock in results[:3]:
                    print(f"  - {stock.get('symbol', 'N/A')}: {stock.get('name', 'N/A')}")
            
            # 测试股票数据获取
            print("\n测试股票数据获取...")
            result = provider.get_stock_data('600519', 'daily', '20240101', '20241026')
            
            # 检查返回结果类型
            if isinstance(result, tuple) and len(result) == 2:
                stock_data, temp_file_path = result
            else:
                # 兼容旧版本，只有股票数据
                stock_data = result
                temp_file_path = None
            
            if stock_data is not None and not stock_data.empty:
                print(f"✓ 成功获取股票数据，数据量: {len(stock_data)}")
                print(f"  列名: {list(stock_data.columns)}")
                print(f"  数据范围: {stock_data.iloc[0]['日期']} 到 {stock_data.iloc[-1]['日期']}")
                if temp_file_path:
                    print(f"  临时文件路径: {temp_file_path}")
            else:
                print("✗ 获取股票数据失败")
                
            return True
                
        except Exception as e:
            print(f"✗ 初始化或测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except ImportError as e:
        print(f"✗ 导入AkshareDataProvider失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_app_data_provider():
    """测试app.py中的数据提供者状态"""
    print("\n" + "=" * 60)
    print("测试app.py中的数据提供者状态")
    print("=" * 60)
    
    try:
        # 模拟app.py中的导入过程
        try:
            from akshare_data_provider import AkshareDataProvider
            data_provider = AkshareDataProvider()
            print("✓ app.py中的数据提供者初始化成功")
            
            # 检查是否为None
            if data_provider is None:
                print("✗ data_provider为None")
                return False
            else:
                print("✓ data_provider不为None")
                
            # 测试功能
            print("\n测试实时股票数据获取...")
            stock_code = "600519"
            end_date_str = "20241026"
            start_date_str = "20231026"  # 1年数据
            
            result = data_provider.get_stock_data(stock_code, 'daily', start_date_str, end_date_str)
            
            # 检查返回结果类型
            if isinstance(result, tuple) and len(result) == 2:
                stock_data, temp_file_path = result
            else:
                # 兼容旧版本，只有股票数据
                stock_data = result
                temp_file_path = None
            
            if stock_data is None or stock_data.empty:
                print("✗ 获取实时股票数据失败")
                return False
            else:
                print(f"✓ 成功获取实时股票数据，数据量: {len(stock_data)}")
                print(f"  列名: {list(stock_data.columns)}")
                if temp_file_path:
                    print(f"  临时文件路径: {temp_file_path}")
                return True
                
        except Exception as e:
            print(f"✗ app.py中的数据提供者测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except ImportError as e:
        print(f"✗ app.py中的数据提供者导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("数据提供者状态诊断")
    print("=" * 60)
    
    # 测试导入
    import_success = test_data_provider_import()
    
    # 测试app.py状态
    app_success = test_app_data_provider()
    
    print("\n" + "=" * 60)
    print("诊断结果:")
    print(f"导入测试: {'通过' if import_success else '失败'}")
    print(f"app.py状态测试: {'通过' if app_success else '失败'}")
    
    if import_success and app_success:
        print("✓ 数据提供者状态正常")
    else:
        print("✗ 数据提供者存在问题，需要修复")
    
    print("=" * 60)

if __name__ == "__main__":
    main()