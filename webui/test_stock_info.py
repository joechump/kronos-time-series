#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试股票信息获取功能
"""

import requests
import json

def test_stock_search():
    """测试股票搜索功能 - 正确的API端点"""
    print("测试股票搜索功能...")
    
    # 测试搜索贵州茅台(600519)
    url = "http://localhost:7070/api/akshare/search-stock"
    
    data = {
        "keyword": "600519"
    }
    
    try:
        response = requests.post(url, json=data)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                stocks = result.get('results', [])
                if stocks:
                    stock = stocks[0]
                    print("股票信息获取成功:")
                    print(f"代码: {stock.get('symbol', 'N/A')}")
                    print(f"名称: {stock.get('name', 'N/A')}")
                    print(f"当前价格: {stock.get('latest_price', 'N/A')}")
                    print(f"涨跌幅: {stock.get('change_rate', 'N/A')}")
                    print(f"涨跌额: {stock.get('change_amount', 'N/A')}")
                    print(f"成交量: {stock.get('volume', 'N/A')}")
                    print(f"成交额: {stock.get('amount', 'N/A')}")
                else:
                    print("未找到匹配的股票")
            else:
                print(f"搜索失败: {result.get('error', '未知错误')}")
        else:
            print(f"HTTP错误: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("连接失败，请确保Web服务器正在运行")
    except Exception as e:
        print(f"测试过程中出现错误: {e}")

def test_stock_data():
    """测试获取股票历史数据"""
    print("\n测试获取股票历史数据...")
    
    url = "http://localhost:7070/api/akshare/get-stock-data"
    
    data = {
        "symbol": "600519",
        "start_date": "20240101",
        "end_date": "20241104",
        "period": "daily"
    }
    
    try:
        response = requests.post(url, json=data)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if 'error' not in result:
                print("股票历史数据获取成功")
                print(f"数据行数: {len(result.get('data', []))}")
                if result.get('data'):
                    first_data = result['data'][0]
                    last_data = result['data'][-1]
                    print(f"最早日期: {first_data.get('date', 'N/A')}")
                    print(f"最晚日期: {last_data.get('date', 'N/A')}")
            else:
                print(f"获取数据失败: {result.get('error', '未知错误')}")
        else:
            print(f"HTTP错误: {response.text}")
            
    except Exception as e:
        print(f"数据获取测试过程中出现错误: {e}")

def test_stock_search_by_name():
    """测试按名称搜索股票"""
    print("\n测试按名称搜索股票...")
    
    url = "http://localhost:7070/api/akshare/search-stock"
    
    data = {
        "keyword": "茅台"
    }
    
    try:
        response = requests.post(url, json=data)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                stocks = result.get('results', [])
                print(f"搜索到 {len(stocks)} 个匹配的股票:")
                for stock in stocks:
                    print(f"  {stock.get('symbol')} - {stock.get('name')}")
            else:
                print(f"搜索失败: {result.get('error', '未知错误')}")
        else:
            print(f"HTTP错误: {response.text}")
            
    except Exception as e:
        print(f"名称搜索测试过程中出现错误: {e}")

if __name__ == "__main__":
    print("=" * 50)
    print("股票信息功能测试")
    print("=" * 50)
    
    test_stock_search()
    test_stock_data()
    test_stock_search_by_name()
    
    print("\n测试完成!")