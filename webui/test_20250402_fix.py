#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试2025年4月2日预测API修复
验证数据不足问题的根本原因
"""

import requests
import json
from datetime import datetime, timedelta

def test_trading_day_check(date_str):
    """测试交易日检查API"""
    url = "http://localhost:8080/api/trading-calendar/check"
    params = {"date": date_str}
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 交易日检查成功")
            print(f"📊 是否为交易日: {data.get('is_trading_day', 'N/A')}")
            return True
        else:
            print(f"❌ 交易日检查失败 (状态码: {response.status_code})")
            print(f"📋 错误信息: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 交易日检查异常: {e}")
        return False

def test_prediction_api(date_str):
    """测试预测API"""
    url = "http://localhost:8080/api/predict"
    
    # 使用不同的股票代码测试
    test_cases = [
        {
            "file_path": "stock_000858_live",  # 五粮液
            "lookback": 120,
            "pred_len": 30,
            "start_date": date_str
        },
        {
            "file_path": "stock_600519_live",  # 贵州茅台
            "lookback": 120,
            "pred_len": 30,
            "start_date": date_str
        },
        {
            "file_path": "stock_000001_live",  # 平安银行
            "lookback": 120,
            "pred_len": 30,
            "start_date": date_str
        }
    ]
    
    for i, params in enumerate(test_cases):
        print(f"\n📋 测试用例 {i+1}: {params['file_path']}")
        
        try:
            response = requests.post(url, json=params)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 预测成功")
                print(f"📊 预测结果: {data.get('prediction', 'N/A')}")
                return True
            else:
                print(f"❌ 预测失败 (状态码: {response.status_code})")
                print(f"📋 错误信息: {response.text}")
                
        except Exception as e:
            print(f"❌ 预测异常: {e}")
    
    return False

def test_data_availability():
    """测试数据可用性"""
    print("\n📊 测试数据可用性")
    
    # 测试不同日期的数据
    test_dates = [
        "2025-04-02",  # 用户提到的日期
        "2025-03-29",  # 之前的测试日期
        "2025-03-28",  # 工作日
        "2025-03-27",  # 更早的日期
    ]
    
    for date_str in test_dates:
        print(f"\n📅 测试日期: {date_str}")
        print("=" * 50)
        
        # 交易日检查
        if test_trading_day_check(date_str):
            # 预测API测试
            test_prediction_api(date_str)

def main():
    """主函数"""
    print("🚀 开始测试2025年4月2日预测API修复")
    print("=" * 60)
    
    # 测试数据可用性
    test_data_availability()
    
    print("\n" + "=" * 60)
    print("📊 测试完成")

if __name__ == "__main__":
    main()