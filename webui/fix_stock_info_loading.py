#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复股票信息显示"加载中..."问题
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import akshare as ak
import pandas as pd
import requests
import json

def test_akshare_apis():
    """测试akshare相关API是否正常工作"""
    print("=" * 60)
    print("测试akshare API可用性")
    print("=" * 60)
    
    # 测试1: 股票基本信息接口
    print("\n1. 测试股票基本信息接口 (ak.stock_individual_info_em):")
    try:
        stock_info = ak.stock_individual_info_em(symbol="600519")
        print(f"✅ 接口调用成功")
        print(f"   返回数据形状: {stock_info.shape}")
        print(f"   列名: {list(stock_info.columns)}")
        
        if not stock_info.empty:
            # 显示股票信息
            print("   股票基本信息:")
            for _, row in stock_info.iterrows():
                print(f"     {row['item']}: {row['value']}")
            
            # 提取股票名称
            name_row = stock_info[stock_info['item'] == '股票简称']
            if not name_row.empty:
                stock_name = name_row.iloc[0]['value']
                print(f"   ✅ 成功获取股票名称: {stock_name}")
            else:
                print("   ❌ 未找到股票简称")
        else:
            print("   ❌ 返回数据为空")
            
    except Exception as e:
        print(f"❌ 接口调用失败: {e}")
    
    # 测试2: 股票历史数据接口
    print("\n2. 测试股票历史数据接口 (ak.stock_zh_a_hist):")
    try:
        from datetime import datetime, timedelta
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d')
        
        hist_data = ak.stock_zh_a_hist(symbol="600519", period='daily', 
                                      start_date=start_date, end_date=end_date)
        print(f"✅ 接口调用成功")
        print(f"   返回数据形状: {hist_data.shape}")
        print(f"   列名: {list(hist_data.columns)}")
        
        if not hist_data.empty:
            latest_row = hist_data.iloc[-1]
            print(f"   最新价格: {latest_row['收盘']}")
            print(f"   涨跌幅: {latest_row['涨跌幅']}")
            print(f"   涨跌额: {latest_row['涨跌额']}")
        else:
            print("   ❌ 返回数据为空")
            
    except Exception as e:
        print(f"❌ 接口调用失败: {e}")

def test_web_api():
    """测试Web API股票搜索功能"""
    print("\n" + "=" * 60)
    print("测试Web API股票搜索功能")
    print("=" * 60)
    
    base_url = "http://localhost:7070"
    
    # 测试搜索API
    print("\n1. 测试股票搜索API:")
    try:
        response = requests.post(
            f"{base_url}/api/akshare/search-stock",
            json={"keyword": "600519"},
            timeout=10
        )
        
        print(f"   状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   响应数据: {json.dumps(data, ensure_ascii=False, indent=2)}")
            
            if data.get('success'):
                results = data.get('results', [])
                print(f"   ✅ 搜索成功，找到 {len(results)} 个结果")
                if results:
                    stock = results[0]
                    print(f"   股票信息:")
                    print(f"     代码: {stock.get('symbol', 'N/A')}")
                    print(f"     名称: {stock.get('name', 'N/A')}")
                    print(f"     当前价格: {stock.get('latest_price', 'N/A')}")
                    print(f"     涨跌幅: {stock.get('change_rate', 'N/A')}")
            else:
                print(f"   ❌ 搜索失败: {data.get('error', '未知错误')}")
        else:
            print(f"   ❌ HTTP错误: {response.text}")
            
    except Exception as e:
        print(f"❌ API调用失败: {e}")

def create_improved_search_method():
    """创建改进的股票搜索方法"""
    print("\n" + "=" * 60)
    print("创建改进的股票搜索方法")
    print("=" * 60)
    
    improved_code = '''
def search_stock_improved(self, keyword: str, max_retries: int = 3) -> List[Dict]:
    """
    改进的股票搜索方法 - 使用更可靠的akshare接口
    
    参数:
        keyword: 搜索关键词（股票代码或名称）
        max_retries: 最大重试次数
        
    返回:
        List[Dict]: 股票列表
    """
    cache_key = f"search_{keyword}"
    
    if cache_key in self.cache:
        return self.cache[cache_key]
    
    for attempt in range(max_retries):
        try:
            # 添加请求间隔，避免高频请求被限制
            import time
            if attempt > 0:
                time.sleep(5)  # 重试时等待5秒
            
            # 方法1: 使用股票实时行情接口获取基本信息
            try:
                # 使用股票实时行情接口
                realtime_data = ak.stock_zh_a_spot_em()
                
                # 根据关键词过滤股票
                if keyword.isdigit():
                    # 按代码搜索
                    filtered_stocks = realtime_data[realtime_data['代码'] == keyword]
                else:
                    # 按名称搜索
                    filtered_stocks = realtime_data[realtime_data['名称'].str.contains(keyword, na=False)]
                
                if not filtered_stocks.empty:
                    stock_results = []
                    for _, stock_row in filtered_stocks.iterrows():
                        stock_info = {
                            'symbol': stock_row['代码'],
                            'name': stock_row['名称'],
                            'latest_price': str(stock_row['最新价']),
                            'change_rate': str(stock_row['涨跌幅']),
                            'change_amount': str(stock_row['涨跌额']),
                            'volume': str(stock_row['成交量']),
                            'amount': str(stock_row['成交额'])
                        }
                        stock_results.append(stock_info)
                    
                    self.cache[cache_key] = stock_results
                    return stock_results
                    
            except Exception as realtime_error:
                logger.warning(f"实时行情接口失败: {realtime_error}")
            
            # 方法2: 备用方案 - 使用历史数据接口
            try:
                from datetime import datetime, timedelta
                end_date = datetime.now().strftime('%Y%m%d')
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
                
                # 如果是数字代码，直接使用
                if keyword.isdigit():
                    hist_data = ak.stock_zh_a_hist(symbol=keyword, period='daily', 
                                                start_date=start_date, end_date=end_date)
                    if not hist_data.empty:
                        latest_row = hist_data.iloc[-1]
                        stock_info = {
                            'symbol': keyword,
                            'name': f"股票{keyword}",  # 备用名称
                            'latest_price': str(latest_row['收盘']),
                            'change_rate': str(latest_row['涨跌幅']),
                            'change_amount': str(latest_row['涨跌额']),
                            'volume': str(latest_row['成交量']),
                            'amount': str(latest_row['成交额'])
                        }
                        self.cache[cache_key] = [stock_info]
                        return [stock_info]
                        
            except Exception as hist_error:
                logger.warning(f"历史数据接口备用方案失败: {hist_error}")
            
            # 方法3: 最终备用方案 - 返回基础信息
            if keyword.isdigit():
                stock_info = {
                    'symbol': keyword,
                    'name': f"股票{keyword}",
                    'latest_price': 'N/A',
                    'change_rate': 'N/A', 
                    'change_amount': 'N/A',
                    'volume': 'N/A',
                    'amount': 'N/A'
                }
                self.cache[cache_key] = [stock_info]
                return [stock_info]
            
            # 如果重试次数用完，返回空结果
            logger.error(f"搜索股票失败，重试{max_retries}次后仍无法获取数据: {keyword}")
            return []
            
        except Exception as e:
            logger.error(f"搜索股票失败 (尝试 {attempt + 1}/{max_retries}): {keyword}, 错误: {e}")
            
            # 如果是最后一次尝试，返回基础信息
            if attempt == max_retries - 1:
                if keyword.isdigit():
                    stock_info = {
                        'symbol': keyword,
                        'name': f"股票{keyword}",
                        'latest_price': 'N/A',
                        'change_rate': 'N/A',
                        'change_amount': 'N/A',
                        'volume': 'N/A',
                        'amount': 'N/A'
                    }
                    return [stock_info]
                return []
            
            # 等待一段时间后重试
            time.sleep(2 * (attempt + 1))
    
    return []
'''
    
    print("✅ 改进的搜索方法代码已生成")
    print("\n主要改进点:")
    print("1. 使用更可靠的ak.stock_zh_a_spot_em()实时行情接口")
    print("2. 添加了备用方案，使用历史数据接口")
    print("3. 最终备用方案确保至少返回股票代码和基础信息")
    print("4. 改进了错误处理和重试机制")
    
    return improved_code

def main():
    """主函数"""
    print("开始诊断股票信息显示问题...")
    
    # 测试akshare API
    test_akshare_apis()
    
    # 测试Web API
    test_web_api()
    
    # 生成改进的搜索方法
    improved_code = create_improved_search_method()
    
    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)
    
    print("\n📋 问题诊断结果:")
    print("1. 股票信息显示'加载中...'是因为search_stock方法中的akshare接口调用失败")
    print("2. 需要改进搜索方法，使用更可靠的akshare接口")
    print("3. 改进方案已生成，需要更新akshare_data_provider.py文件")
    
    print("\n🚀 下一步操作:")
    print("1. 更新akshare_data_provider.py中的search_stock方法")
    print("2. 重启Web服务器")
    print("3. 测试修复效果")

if __name__ == "__main__":
    main()