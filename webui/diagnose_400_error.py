#!/usr/bin/env python3
"""
诊断预测API 400错误问题
直接测试后端API，找出导致400错误的具体原因
"""

import requests
import json
import sys

def test_api_with_different_params():
    """测试不同参数组合，找出导致400错误的原因"""
    
    base_url = "http://localhost:8080"
    
    # 测试用例列表
    test_cases = [
        {
            "name": "测试1: 使用前端设置的currentDataFile格式",
            "params": {
                "file_path": "stock_600519_live",
                "lookback": 100,
                "pred_len": 30,
                "start_date": None,
                "temperature": 1.3,
                "top_p": 0.98,
                "sample_count": 2
            }
        },
        {
            "name": "测试2: 使用完整的文件路径格式",
            "params": {
                "file_path": "stock_600519_live.csv",
                "lookback": 100,
                "pred_len": 30,
                "start_date": None,
                "temperature": 1.3,
                "top_p": 0.98,
                "sample_count": 2
            }
        },
        {
            "name": "测试3: 使用绝对路径格式",
            "params": {
                "file_path": "c:\\kron\\data\\stock_600519_live.csv",
                "lookback": 100,
                "pred_len": 30,
                "start_date": None,
                "temperature": 1.3,
                "top_p": 0.98,
                "sample_count": 2
            }
        },
        {
            "name": "测试4: 使用相对路径格式",
            "params": {
                "file_path": "data/stock_600519_live.csv",
                "lookback": 100,
                "pred_len": 30,
                "start_date": None,
                "temperature": 1.3,
                "top_p": 0.98,
                "sample_count": 2
            }
        },
        {
            "name": "测试5: 使用空字符串作为文件路径",
            "params": {
                "file_path": "",
                "lookback": 100,
                "pred_len": 30,
                "start_date": None,
                "temperature": 1.3,
                "top_p": 0.98,
                "sample_count": 2
            }
        },
        {
            "name": "测试6: 使用None作为文件路径",
            "params": {
                "file_path": None,
                "lookback": 100,
                "pred_len": 30,
                "start_date": None,
                "temperature": 1.3,
                "top_p": 0.98,
                "sample_count": 2
            }
        },
        {
            "name": "测试7: 使用data_file参数名（前端可能使用的参数名）",
            "params": {
                "data_file": "stock_600519_live",
                "lookback": 100,
                "pred_len": 30,
                "start_date": None,
                "temperature": 1.3,
                "top_p": 0.98,
                "sample_count": 2
            }
        }
    ]
    
    print("🔍 开始诊断预测API 400错误问题...")
    print("=" * 80)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 {test_case['name']}")
        print("-" * 60)
        
        try:
            # 发送POST请求
            response = requests.post(
                f"{base_url}/api/predict",
                json=test_case['params'],
                timeout=30
            )
            
            print(f"   📤 请求参数: {json.dumps(test_case['params'], indent=2, ensure_ascii=False)}")
            print(f"   📥 响应状态: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ 请求成功: {result.get('message', '预测成功')}")
                if result.get('success'):
                    print(f"   📊 预测结果: {result.get('prediction', '无结果')}")
            elif response.status_code == 400:
                error_data = response.json()
                print(f"   ❌ 400错误: {error_data.get('error', '未知错误')}")
                print(f"   🔍 错误详情: {error_data}")
            else:
                print(f"   ⚠️ 其他错误: {response.status_code}")
                print(f"   📝 响应内容: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"   💥 请求异常: {e}")
        except Exception as e:
            print(f"   💥 其他异常: {e}")
    
    print("\n" + "=" * 80)
    print("🔍 诊断完成")

def check_backend_api_structure():
    """检查后端API的结构和期望的参数格式"""
    
    print("\n🔍 检查后端API结构...")
    print("-" * 60)
    
    base_url = "http://localhost:8080"
    
    # 检查API文档或端点信息
    try:
        # 尝试获取API信息
        response = requests.get(f"{base_url}/api/model-status", timeout=10)
        if response.status_code == 200:
            model_status = response.json()
            print(f"   ✅ 模型状态API正常")
            print(f"   📊 模型信息: {model_status}")
        else:
            print(f"   ❌ 模型状态API异常: {response.status_code}")
    except Exception as e:
        print(f"   💥 检查模型状态失败: {e}")
    
    # 检查是否有API文档端点
    try:
        response = requests.get(f"{base_url}/api/docs", timeout=10)
        if response.status_code == 200:
            print(f"   ✅ API文档端点存在")
        else:
            print(f"   ❌ API文档端点不存在: {response.status_code}")
    except:
        print(f"   ❌ API文档端点访问失败")

def main():
    """主函数"""
    
    print("🔍 Kronos 预测API 400错误诊断工具")
    print("=" * 80)
    
    # 检查后端API结构
    check_backend_api_structure()
    
    # 测试不同参数组合
    test_api_with_different_params()
    
    print("\n📋 诊断总结:")
    print("-" * 60)
    print("1. 通过测试不同参数组合，找出导致400错误的具体原因")
    print("2. 确定后端API期望的文件路径格式")
    print("3. 提供修复建议")

if __name__ == "__main__":
    main()