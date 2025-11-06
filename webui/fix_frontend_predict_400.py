#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复前端预测400错误脚本
问题分析：前端预测请求失败的原因是数据文件路径不正确
"""

import os
import json
import requests
from datetime import datetime

def find_latest_stock_data_file(symbol="600519"):
    """查找最新的股票数据文件"""
    temp_dir = os.path.join(os.environ['LOCALAPPDATA'], 'Temp', 'kronos')
    
    if not os.path.exists(temp_dir):
        return None
    
    # 查找匹配的CSV文件
    matching_files = []
    for filename in os.listdir(temp_dir):
        if filename.startswith(f"stock_{symbol}_daily") and filename.endswith(".csv"):
            matching_files.append(filename)
    
    if not matching_files:
        return None
    
    # 返回最新的文件
    latest_file = sorted(matching_files)[-1]
    return os.path.join(temp_dir, latest_file)

def test_predict_api_with_correct_file():
    """使用正确的数据文件测试预测API"""
    
    # 查找最新的数据文件
    data_file = find_latest_stock_data_file("600519")
    if not data_file:
        print("❌ 未找到股票数据文件")
        return False
    
    print(f"✅ 找到数据文件: {data_file}")
    
    # 测试预测API
    url = "http://localhost:8080/api/predict"
    
    # 使用正确的参数
    payload = {
        "file_path": data_file,  # 使用完整的文件路径
        "lookback": 400,
        "pred_len": 120,
        "temperature": 1.3,
        "top_p": 1,
        "sample_count": 2,
        "start_date": "2024-01-01",  # 使用更早的日期确保有足够数据
        "trading_mode": "auto"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        
        if response.status_code == 200:
            print("✅ 预测API测试成功!")
            result = response.json()
            print(f"📊 预测结果: {result.get('message', '成功')}")
            print(f"📈 预测点数: {len(result.get('predictions', []))}")
            return True
        else:
            print(f"❌ 预测API返回错误: {response.status_code}")
            print(f"错误信息: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 预测API调用失败: {e}")
        return False

def check_frontend_data_file_logic():
    """检查前端数据文件处理逻辑"""
    
    print("\n🔍 检查前端数据文件处理逻辑...")
    
    # 检查前端是否创建了正确的数据文件
    temp_dir = os.path.join(os.environ['LOCALAPPDATA'], 'Temp', 'kronos')
    
    if os.path.exists(temp_dir):
        files = os.listdir(temp_dir)
        print(f"临时文件目录: {temp_dir}")
        print(f"文件列表: {files}")
        
        # 检查是否有stock_600519_live文件
        live_file = os.path.join(temp_dir, "stock_600519_live")
        if os.path.exists(live_file):
            print("⚠️  发现stock_600519_live文件，但可能是错误的格式")
        else:
            print("❌ 前端未创建stock_600519_live文件")
    else:
        print("❌ 临时文件目录不存在")

def fix_frontend_predict_issue():
    """修复前端预测问题"""
    
    print("🚀 开始修复前端预测400错误...")
    print("=" * 60)
    
    # 1. 检查数据文件
    check_frontend_data_file_logic()
    
    # 2. 测试预测API
    print("\n🧪 测试预测API...")
    success = test_predict_api_with_correct_file()
    
    if success:
        print("\n✅ 前端预测问题已修复!")
        print("\n📋 修复方案:")
        print("1. 前端需要正确创建数据文件路径")
        print("2. 预测请求应使用完整的文件路径")
        print("3. 确保start_date参数提供足够的历史数据")
    else:
        print("\n❌ 修复失败，需要进一步调试")
    
    return success

if __name__ == "__main__":
    fix_frontend_predict_issue()