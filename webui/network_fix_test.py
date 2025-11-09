#!/usr/bin/env python3
"""
网络连接问题诊断和修复测试
"""

import requests
import socket
import urllib3
from datetime import datetime

def test_dns_resolution():
    """测试DNS解析"""
    print("=== DNS解析测试 ===")
    
    test_domains = [
        'push2his.eastmoney.com',
        '82.push2.eastmoney.com', 
        'www.baidu.com',
        'www.google.com'
    ]
    
    for domain in test_domains:
        try:
            ip = socket.gethostbyname(domain)
            print(f"✅ {domain} -> {ip}")
        except Exception as e:
            print(f"❌ {domain} DNS解析失败: {e}")

def test_http_connectivity():
    """测试HTTP连接"""
    print("\n=== HTTP连接测试 ===")
    
    test_urls = [
        'http://www.baidu.com',
        'https://www.baidu.com',
        'https://push2his.eastmoney.com',
        'https://82.push2.eastmoney.com'
    ]
    
    for url in test_urls:
        try:
            response = requests.get(url, timeout=10)
            print(f"✅ {url} -> 状态码: {response.status_code}")
        except Exception as e:
            print(f"❌ {url} 连接失败: {e}")

def test_alternative_data_sources():
    """测试备用数据源"""
    print("\n=== 备用数据源测试 ===")
    
    # 测试其他可能的数据源
    alternative_sources = [
        'https://api.finance.com',
        'https://stock.api.com',
        'https://data.example.com'
    ]
    
    for source in alternative_sources:
        try:
            response = requests.get(source, timeout=5)
            print(f"✅ {source} -> 状态码: {response.status_code}")
        except Exception as e:
            print(f"❌ {source} 不可用: {e}")

def test_local_fallback():
    """测试本地备用方案"""
    print("\n=== 本地备用方案测试 ===")
    
    # 测试使用本地文件或模拟数据
    try:
        # 创建模拟数据
        import pandas as pd
        import numpy as np
        
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        data = {
            'timestamps': dates,
            'open': np.random.normal(1000, 50, len(dates)),
            'high': np.random.normal(1050, 50, len(dates)), 
            'low': np.random.normal(950, 50, len(dates)),
            'close': np.random.normal(1020, 50, len(dates)),
            'volume': np.random.randint(1000000, 5000000, len(dates))
        }
        
        df = pd.DataFrame(data)
        print(f"✅ 模拟数据创建成功: {len(df)} 条记录")
        print(f"数据列: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        print(f"❌ 模拟数据创建失败: {e}")
        return None

def main():
    """主测试函数"""
    print("网络连接问题诊断和修复测试")
    print("=" * 50)
    
    test_dns_resolution()
    test_http_connectivity()
    test_alternative_data_sources()
    
    # 测试本地备用方案
    df = test_local_fallback()
    
    print("\n=== 修复建议 ===")
    print("1. 检查网络连接和DNS设置")
    print("2. 实现本地模拟数据备用方案")
    print("3. 添加重试机制和超时设置")
    print("4. 提供更友好的错误信息")

if __name__ == "__main__":
    main()