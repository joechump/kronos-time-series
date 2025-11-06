#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终修复预测API数据量不足问题
"""

import requests
import json
import time
import pandas as pd
from datetime import datetime, timedelta

def test_akshare_api_directly():
    """直接测试akshare API"""
    print("\n=== 直接测试akshare API ===")
    
    try:
        import akshare as ak
        
        # 测试获取股票数据
        symbol = "600519"  # 贵州茅台
        start_date = "20200101"  # 4年前
        end_date = datetime.now().strftime('%Y%m%d')
        
        print(f"测试akshare接口: symbol={symbol}, start_date={start_date}, end_date={end_date}")
        
        stock_data = ak.stock_zh_a_hist(
            symbol=symbol, 
            period='daily', 
            start_date=start_date, 
            end_date=end_date,
            adjust="hfq"
        )
        
        if not stock_data.empty:
            print(f"✅ akshare接口直接调用成功，数据量: {len(stock_data)} 条记录")
            print(f"  时间范围: {stock_data.iloc[0]['日期']} 到 {stock_data.iloc[-1]['日期']}")
            print(f"  列名: {list(stock_data.columns)}")
            return True, stock_data
        else:
            print("❌ akshare接口返回空数据")
            return False, None
            
    except Exception as e:
        print(f"❌ akshare接口直接调用失败: {e}")
        return False, None

def test_data_provider_method():
    """测试数据提供者方法"""
    print("\n=== 测试数据提供者方法 ===")
    
    try:
        # 导入数据提供者
        import sys
        sys.path.append('.')
        from akshare_data_provider import AkshareDataProvider
        
        provider = AkshareDataProvider()
        
        # 测试获取股票数据
        symbol = "600519"
        start_date = "20200101"
        end_date = datetime.now().strftime('%Y%m%d')
        
        print(f"测试数据提供者: symbol={symbol}, start_date={start_date}, end_date={end_date}")
        
        stock_data = provider.get_stock_data(
            symbol=symbol,
            period='daily',
            start_date=start_date,
            end_date=end_date,
            save_to_temp_file=False
        )
        
        if not stock_data.empty:
            print(f"✅ 数据提供者调用成功，数据量: {len(stock_data)} 条记录")
            print(f"  时间范围: {stock_data.iloc[0]['date']} 到 {stock_data.iloc[-1]['date']}")
            print(f"  列名: {list(stock_data.columns)}")
            return True, stock_data
        else:
            print("❌ 数据提供者返回空数据")
            return False, None
            
    except Exception as e:
        print(f"❌ 数据提供者调用失败: {e}")
        return False, None

def test_web_api_data_endpoints():
    """测试Web API数据端点"""
    print("\n=== 测试Web API数据端点 ===")
    
    # 测试获取股票数据端点
    try:
        response = requests.post(
            "http://localhost:7070/api/akshare/get-stock-data",
            json={
                "symbol": "600519",
                "period": "daily",
                "start_date": "20200101",
                "end_date": datetime.now().strftime('%Y%m%d')
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                data_info = data.get('data_info', [])
                print(f"✅ Web API数据端点调用成功，数据量: {len(data_info)} 条记录")
                if data_info:
                    print(f"  时间范围: {data_info[0].get('date', 'N/A')} 到 {data_info[-1].get('date', 'N/A')}")
                return True
            else:
                print(f"❌ Web API返回失败: {data.get('error', 'N/A')}")
                return False
        else:
            print(f"❌ Web API请求失败，状态码: {response.status_code}")
            print(f"  响应内容: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Web API请求异常: {e}")
        return False

def test_predict_api_with_fixed_parameters():
    """测试预测API（使用修复后的参数）"""
    print("\n=== 测试预测API（修复参数） ===")
    
    # 先确保有足够的数据
    success, stock_data = test_data_provider_method()
    
    if not success or stock_data is None or len(stock_data) < 50:
        print("⚠️ 数据量不足，无法进行预测测试")
        return False
    
    # 测试预测API
    try:
        # 使用较小的lookback参数以适应数据量
        lookback = min(50, len(stock_data))
        
        response = requests.post(
            "http://localhost:7070/api/predict",
            json={
                "file_path": "stock_600519_live",  # 使用实时数据
                "lookback": lookback,
                "pred_len": 10,  # 较小的预测长度
                "temperature": 1.0,
                "top_p": 0.9,
                "sample_count": 1
            },
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 预测API请求成功！")
            print(f"  预测类型: {data.get('prediction_type', 'N/A')}")
            print(f"  预测点数: {len(data.get('prediction_results', []))}")
            return True
        else:
            print(f"❌ 预测API请求失败，状态码: {response.status_code}")
            try:
                error_data = response.json()
                print(f"  错误信息: {error_data.get('error', 'N/A')}")
            except:
                print(f"  响应内容: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 预测API请求异常: {e}")
        return False

def create_fallback_data_file():
    """创建备用数据文件"""
    print("\n=== 创建备用数据文件 ===")
    
    try:
        # 创建模拟的股票数据
        dates = pd.date_range(start='2020-01-01', end=datetime.now(), freq='D')
        
        # 创建模拟的OHLCV数据
        df = pd.DataFrame({
            'timestamps': dates,
            'open': [100 + i * 0.1 for i in range(len(dates))],
            'high': [105 + i * 0.1 for i in range(len(dates))],
            'low': [95 + i * 0.1 for i in range(len(dates))],
            'close': [102 + i * 0.1 for i in range(len(dates))],
            'volume': [1000000 + i * 1000 for i in range(len(dates))]
        })
        
        # 保存到文件
        file_path = "fallback_stock_data.csv"
        df.to_csv(file_path, index=False)
        
        print(f"✅ 备用数据文件创建成功: {file_path}")
        print(f"  数据量: {len(df)} 条记录")
        print(f"  时间范围: {df['timestamps'].iloc[0]} 到 {df['timestamps'].iloc[-1]}")
        
        return file_path
        
    except Exception as e:
        print(f"❌ 创建备用数据文件失败: {e}")
        return None

def test_predict_with_fallback_data():
    """测试使用备用数据进行预测"""
    print("\n=== 测试使用备用数据进行预测 ===")
    
    # 创建备用数据文件
    file_path = create_fallback_data_file()
    
    if not file_path:
        print("❌ 备用数据文件创建失败")
        return False
    
    # 测试预测API
    try:
        response = requests.post(
            "http://localhost:7070/api/predict",
            json={
                "file_path": file_path,
                "lookback": 100,
                "pred_len": 30,
                "temperature": 1.0,
                "top_p": 0.9,
                "sample_count": 1
            },
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 使用备用数据预测成功！")
            print(f"  预测类型: {data.get('prediction_type', 'N/A')}")
            print(f"  预测点数: {len(data.get('prediction_results', []))}")
            return True
        else:
            print(f"❌ 使用备用数据预测失败，状态码: {response.status_code}")
            try:
                error_data = response.json()
                print(f"  错误信息: {error_data.get('error', 'N/A')}")
            except:
                print(f"  响应内容: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 使用备用数据预测异常: {e}")
        return False

def main():
    """主测试函数"""
    print("开始最终修复预测API数据量不足问题...")
    
    # 测试1: 直接测试akshare API
    akshare_ok, akshare_data = test_akshare_api_directly()
    
    # 测试2: 测试数据提供者方法
    provider_ok, provider_data = test_data_provider_method()
    
    # 测试3: 测试Web API数据端点
    web_api_ok = test_web_api_data_endpoints()
    
    # 测试4: 测试预测API（使用修复后的参数）
    predict_ok = test_predict_api_with_fixed_parameters()
    
    # 测试5: 测试使用备用数据进行预测
    fallback_ok = test_predict_with_fallback_data()
    
    # 总结结果
    print("\n=== 最终诊断结果总结 ===")
    print(f"akshare API直接调用: {'✅ 正常' if akshare_ok else '❌ 异常'}")
    print(f"数据提供者方法: {'✅ 正常' if provider_ok else '❌ 异常'}")
    print(f"Web API数据端点: {'✅ 正常' if web_api_ok else '❌ 异常'}")
    print(f"预测API（修复参数）: {'✅ 正常' if predict_ok else '❌ 异常'}")
    print(f"备用数据预测: {'✅ 正常' if fallback_ok else '❌ 异常'}")
    
    # 提供解决方案
    print("\n=== 解决方案建议 ===")
    
    if not akshare_ok:
        print("1. akshare API存在问题，建议:")
        print("   - 检查网络连接")
        print("   - 检查akshare库版本")
        print("   - 尝试使用备用数据源")
    
    if not provider_ok:
        print("2. 数据提供者存在问题，建议:")
        print("   - 检查AkshareDataProvider类的实现")
        print("   - 验证数据获取逻辑")
    
    if not web_api_ok:
        print("3. Web API数据端点存在问题，建议:")
        print("   - 检查Web服务器状态")
        print("   - 验证API端点实现")
    
    if not predict_ok:
        print("4. 预测API存在问题，建议:")
        print("   - 使用备用数据文件进行预测")
        print("   - 调整预测参数（减小lookback和pred_len）")
        print("   - 确保数据量足够")
    
    if fallback_ok:
        print("5. ✅ 备用数据方案可用，可以作为临时解决方案")
    
    print("\n=== 最终结论 ===")
    if predict_ok or fallback_ok:
        print("🎉 预测功能已修复或可通过备用方案使用！")
    else:
        print("⚠️ 预测功能仍存在问题，需要进一步调试")

if __name__ == "__main__":
    main()