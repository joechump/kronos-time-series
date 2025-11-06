#!/usr/bin/env python3
"""
测试Web API股票信息获取功能
"""

import requests
import json

def test_web_api():
    """测试Web API股票信息获取功能"""
    base_url = "http://localhost:7070"
    
    print("=" * 50)
    print("测试Web API股票信息获取功能")
    print("=" * 50)
    
    # 测试1: 搜索股票
    print("\n1. 测试股票搜索API:")
    try:
        response = requests.post(
            f"{base_url}/api/akshare/search-stock",
            json={"keyword": "600519"}
        )
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"搜索结果: {json.dumps(data, ensure_ascii=False, indent=2)}")
        else:
            print(f"错误响应: {response.text}")
    except Exception as e:
        print(f"请求失败: {e}")
    
    # 测试2: 获取股票历史数据
    print("\n2. 测试股票历史数据API:")
    try:
        response = requests.post(
            f"{base_url}/api/akshare/get-stock-data",
            json={
                "symbol": "600519",
                "period": "daily",
                "start_date": "20241101",
                "end_date": "20241105"
            }
        )
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"数据行数: {len(data.get('data', []))}")
            if data.get('data'):
                print("前3条数据:")
                for i, item in enumerate(data['data'][:3]):
                    print(f"  第{i+1}条: {item}")
        else:
            print(f"错误响应: {response.text}")
    except Exception as e:
        print(f"请求失败: {e}")
    
    # 测试3: 测试主页面股票信息显示
    print("\n3. 测试主页面股票信息显示:")
    try:
        response = requests.get(f"{base_url}/")
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            print("主页面加载成功")
            # 检查页面中是否包含股票信息相关元素
            if "600519" in response.text or "贵州茅台" in response.text:
                print("页面中包含股票信息")
            else:
                print("页面中未找到股票信息")
        else:
            print(f"错误响应: {response.text}")
    except Exception as e:
        print(f"请求失败: {e}")
    
    print("\n测试完成!")

if __name__ == "__main__":
    test_web_api()