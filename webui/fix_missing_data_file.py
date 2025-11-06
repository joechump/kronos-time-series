#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复缺失的数据文件问题
创建前端需要的 stock_600519_live.csv 文件
"""

import os
import shutil
import requests
import json
from datetime import datetime

def check_and_fix_data_file():
    """检查和修复数据文件"""
    print("🔧 检查和修复数据文件")
    
    # 数据文件路径
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "downloaded_data")
    target_file = os.path.join(data_dir, "stock_600519_live.csv")
    
    print(f"目标文件: {target_file}")
    
    # 检查目标文件是否存在
    if os.path.exists(target_file):
        print("✅ 目标文件已存在")
        return True
    
    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        print("❌ 数据目录不存在，创建目录")
        os.makedirs(data_dir)
    
    # 查找可用的历史数据文件
    print("📂 查找可用的历史数据文件...")
    available_files = []
    
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.startswith("stock_600519") and file.endswith(".csv"):
                available_files.append(file)
                print(f"  找到文件: {file}")
    
    if not available_files:
        print("❌ 没有找到任何历史数据文件")
        return False
    
    # 选择最新的文件
    latest_file = sorted(available_files)[-1]  # 按文件名排序，取最新的
    source_file = os.path.join(data_dir, latest_file)
    
    print(f"📋 选择最新文件: {latest_file}")
    
    # 复制文件
    try:
        shutil.copy2(source_file, target_file)
        print(f"✅ 成功复制文件: {latest_file} → stock_600519_live.csv")
        
        # 验证文件
        if os.path.exists(target_file):
            file_size = os.path.getsize(target_file)
            print(f"✅ 文件创建成功，大小: {file_size} 字节")
            return True
        else:
            print("❌ 文件复制失败")
            return False
    except Exception as e:
        print(f"❌ 文件复制异常: {e}")
        return False

def test_api_with_fixed_file():
    """测试修复后的API"""
    print("\n🔍 测试修复后的API")
    
    url = "http://localhost:8080/api/predict"
    params = {
        "file_path": "stock_600519_live",
        "lookback": 100,
        "pred_len": 30,
        "model_name": "kronos-small",
        "stock_code": "600519"
    }
    
    try:
        response = requests.post(url, json=params)
        print(f"API响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            predictions = result.get('predictions', [])
            print(f"✅ API测试成功，预测点数: {len(predictions)}")
            return True
        else:
            print(f"❌ API测试失败: {response.text}")
            return False
    except Exception as e:
        print(f"❌ API测试异常: {e}")
        return False

def simulate_frontend_workflow():
    """模拟前端完整工作流程"""
    print("\n🔄 模拟前端完整工作流程")
    
    # 1. 加载股票数据（模拟）
    print("1. 加载股票数据...")
    
    # 2. 检查数据文件
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "downloaded_data")
    target_file = os.path.join(data_dir, "stock_600519_live.csv")
    
    if os.path.exists(target_file):
        print("✅ 数据文件存在")
    else:
        print("❌ 数据文件不存在")
        return False
    
    # 3. 检查模型状态
    print("2. 检查模型状态...")
    try:
        response = requests.get("http://localhost:8080/api/model-status")
        if response.status_code == 200:
            model_status = response.json()
            print(f"✅ 模型状态: {model_status.get('status', 'unknown')}")
        else:
            print("❌ 模型状态检查失败")
            return False
    except Exception as e:
        print(f"❌ 模型状态检查异常: {e}")
        return False
    
    # 4. 执行预测
    print("3. 执行预测...")
    return test_api_with_fixed_file()

def main():
    """主修复函数"""
    print("🔧 修复缺失的数据文件问题")
    print("=" * 50)
    
    # 修复数据文件
    fix_result = check_and_fix_data_file()
    
    if fix_result:
        print("\n✅ 数据文件修复成功")
        
        # 测试API
        api_result = test_api_with_fixed_file()
        
        if api_result:
            print("\n🎉 修复完成！前端现在应该可以正常工作了")
            
            # 模拟前端工作流程
            print("\n🔄 验证前端工作流程...")
            workflow_result = simulate_frontend_workflow()
            
            if workflow_result:
                print("\n✅ 前端工作流程验证成功！")
                print("\n💡 现在请在前端页面：")
                print("1. 重新加载页面（清除缓存）")
                print("2. 输入股票代码 600519")
                print("3. 点击'加载股票数据'按钮")
                print("4. 等待数据加载完成后点击'开始预测'")
            else:
                print("\n⚠️ 前端工作流程验证失败，但数据文件已修复")
        else:
            print("\n❌ API测试失败，请检查后端服务")
    else:
        print("\n❌ 数据文件修复失败")
        print("\n💡 建议：")
        print("1. 在前端页面重新加载股票数据")
        print("2. 检查网络连接")
        print("3. 检查akshare数据源")

if __name__ == "__main__":
    main()