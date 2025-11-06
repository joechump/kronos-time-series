#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单预测功能测试脚本
直接测试预测API，检查400错误具体原因
"""

import requests
import json
import time

def test_predict_api():
    """测试预测API"""
    
    print("🔍 开始测试预测API...")
    
    # 测试数据
    test_data = {
        "file_path": "stock_600519_live",
        "prediction_points": 30,
        "model_name": "kronos-small"
    }
    
    print(f"📦 请求数据: {json.dumps(test_data, indent=2, ensure_ascii=False)}")
    
    try:
        # 发送预测请求
        response = requests.post(
            'http://localhost:8080/api/predict',
            json=test_data,
            timeout=30
        )
        
        print(f"📡 响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ 预测API请求成功！")
            result = response.json()
            print(f"📊 预测结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
            
            # 检查预测点数
            if 'prediction_points' in result:
                print(f"🎯 预测点数: {result['prediction_points']}")
            else:
                print("⚠️ 预测结果中未找到预测点数")
                
        else:
            print(f"❌ 预测API请求失败: {response.status_code}")
            print(f"💥 错误详情: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"💥 请求异常: {e}")
    except Exception as e:
        print(f"💥 其他异常: {e}")

def test_frontend_workflow():
    """测试前端工作流程"""
    
    print("\n🔄 测试前端工作流程...")
    
    # 1. 检查模型状态
    print("1️⃣ 检查模型状态...")
    try:
        response = requests.get('http://localhost:8080/api/model-status', timeout=10)
        if response.status_code == 200:
            model_status = response.json()
            print(f"✅ 模型状态: {json.dumps(model_status, indent=2, ensure_ascii=False)}")
        else:
            print(f"❌ 模型状态检查失败: {response.status_code}")
    except Exception as e:
        print(f"💥 模型状态检查异常: {e}")
    
    # 2. 检查数据文件
    print("\n2️⃣ 检查数据文件...")
    data_files = [
        "stock_600519_live.csv",
        "stock_600519_live"
    ]
    
    for file_name in data_files:
        file_path = f"c:\\kron\\webui\\{file_name}"
        if os.path.exists(file_path):
            print(f"✅ 数据文件存在: {file_name}")
            # 检查文件大小
            file_size = os.path.getsize(file_path)
            print(f"   📏 文件大小: {file_size} 字节")
        else:
            print(f"❌ 数据文件不存在: {file_name}")
    
    # 3. 模拟前端参数
    print("\n3️⃣ 模拟前端参数...")
    
    # 前端可能发送的参数
    frontend_params = {
        "file_path": "stock_600519_live",
        "prediction_points": 30,
        "model_name": "kronos-small"
    }
    
    print(f"📋 前端参数: {json.dumps(frontend_params, indent=2, ensure_ascii=False)}")
    
    # 4. 测试不同参数组合
    print("\n4️⃣ 测试不同参数组合...")
    
    test_cases = [
        {
            "name": "标准参数",
            "params": {"file_path": "stock_600519_live", "prediction_points": 30, "model_name": "kronos-small"}
        },
        {
            "name": "不带模型名",
            "params": {"file_path": "stock_600519_live", "prediction_points": 30}
        },
        {
            "name": "不同预测点数",
            "params": {"file_path": "stock_600519_live", "prediction_points": 10}
        }
    ]
    
    for test_case in test_cases:
        print(f"\n🧪 测试: {test_case['name']}")
        print(f"   📦 参数: {test_case['params']}")
        
        try:
            response = requests.post(
                'http://localhost:8080/api/predict',
                json=test_case['params'],
                timeout=30
            )
            
            print(f"   📡 状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ 成功 - 预测点数: {result.get('prediction_points', 'N/A')}")
            else:
                print(f"   ❌ 失败 - 错误: {response.text}")
                
        except Exception as e:
            print(f"   💥 异常: {e}")

def check_server_status():
    """检查服务器状态"""
    
    print("\n🌐 检查服务器状态...")
    
    try:
        response = requests.get('http://localhost:8080/', timeout=5)
        if response.status_code == 200:
            print("✅ 服务器运行正常")
        else:
            print(f"⚠️ 服务器响应异常: {response.status_code}")
    except Exception as e:
        print(f"❌ 服务器连接失败: {e}")

def main():
    """主函数"""
    
    print("🔍 Kronos 预测功能测试")
    print("=" * 50)
    
    # 检查服务器状态
    check_server_status()
    
    # 测试预测API
    test_predict_api()
    
    # 测试前端工作流程
    test_frontend_workflow()
    
    print("\n🎯 测试完成！")
    print("💡 请根据测试结果分析400错误原因")

if __name__ == "__main__":
    import os
    main()