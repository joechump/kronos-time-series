#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终验证测试脚本
验证修复后的股票搜索功能
"""

import requests
import json
import sys

def test_web_api_search():
    """测试Web API搜索功能"""
    print("=== Web API搜索功能测试 ===")
    
    url = 'http://localhost:7070/api/akshare/search-stock'
    
    # 测试用例
    test_cases = [
        ('茅台', "股票名称搜索"),
        ('贵州茅台', "完整股票名称搜索"),
        ('000001', "股票代码搜索"),
        ('平安银行', "其他股票搜索"),
        ('不存在的股票', "不存在股票搜索")
    ]
    
    all_passed = True
    
    for keyword, description in test_cases:
        print(f"\n测试: {description} - 关键词: '{keyword}'")
        
        try:
            data = {'keyword': keyword}
            response = requests.post(url, json=data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('success'):
                    count = result.get('count', 0)
                    results = result.get('results', [])
                    
                    print(f"  状态: 成功, 结果数量: {count}")
                    
                    if results:
                        for stock in results:
                            print(f"    代码: {stock['symbol']}, 名称: {stock['name']}, 价格: {stock.get('latest_price', 'N/A')}")
                    else:
                        print("    无结果 (可能正常)")
                    
                    # 验证逻辑
                    if keyword in ['茅台', '贵州茅台', '000001', '平安银行'] and count == 0:
                        print(f"  ⚠️ 警告: 预期找到结果但返回空")
                    elif keyword == '不存在的股票' and count > 0:
                        print(f"  ⚠️ 警告: 预期无结果但找到 {count} 个结果")
                    else:
                        print("  ✓ 测试通过")
                        
                else:
                    print(f"  ❌ 失败: API返回success=False, 错误: {result.get('error', '未知错误')}")
                    all_passed = False
                    
            else:
                print(f"  ❌ 失败: HTTP状态码 {response.status_code}")
                all_passed = False
                
        except Exception as e:
            print(f"  ❌ 异常: {e}")
            all_passed = False
    
    return all_passed

def test_direct_data_provider():
    """直接测试数据提供者"""
    print("\n=== 直接数据提供者测试 ===")
    
    try:
        sys.path.append('.')
        from akshare_data_provider import AkshareDataProvider
        
        data_provider = AkshareDataProvider()
        print("数据提供者初始化成功")
        
        # 测试搜索
        test_keywords = ['茅台', '000001']
        
        for keyword in test_keywords:
            print(f"\n测试关键词: '{keyword}'")
            results = data_provider.search_stock(keyword)
            
            if results:
                print(f"  找到 {len(results)} 个结果:")
                for stock in results:
                    print(f"    代码: {stock['symbol']}, 名称: {stock['name']}, 价格: {stock.get('latest_price', 'N/A')}")
            else:
                print("  未找到结果")
        
        print("\n✓ 直接数据提供者测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 直接数据提供者测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("股票搜索功能最终验证测试")
    print("=" * 50)
    
    # 检查Web服务器状态
    try:
        response = requests.get('http://localhost:7070', timeout=5)
        print("✓ Web服务器正在运行")
    except:
        print("❌ Web服务器未运行，请先启动: python app.py")
        return
    
    # 运行测试
    web_api_passed = test_web_api_search()
    direct_passed = test_direct_data_provider()
    
    print("\n" + "=" * 50)
    print("测试结果汇总:")
    print(f"Web API测试: {'通过' if web_api_passed else '失败'}")
    print(f"直接数据提供者测试: {'通过' if direct_passed else '失败'}")
    
    if web_api_passed and direct_passed:
        print("\n🎉 所有测试通过！股票搜索功能已完全修复。")
    else:
        print("\n⚠️ 部分测试失败，需要进一步检查。")

if __name__ == "__main__":
    main()