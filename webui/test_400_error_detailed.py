#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细测试预测API 400错误
模拟前端发送的预测请求，捕获具体的错误信息
"""

import requests
import json
import sys

def test_prediction_api_with_detailed_error():
    """测试预测API并获取详细错误信息"""
    
    print("=" * 80)
    print("🔍 详细测试预测API 400错误")
    print("=" * 80)
    
    # 测试不同的参数组合，模拟前端可能发送的请求
    test_cases = [
        {
            "name": "测试1: 正常股票代码预测",
            "params": {
                "file_path": "stock_600519_live",
                "lookback": 100,
                "pred_len": 30,
                "temperature": 1.0,
                "top_p": 0.9,
                "sample_count": 1
            }
        },
        {
            "name": "测试2: 带start_date参数",
            "params": {
                "file_path": "stock_600519_live",
                "lookback": 100,
                "pred_len": 30,
                "temperature": 1.0,
                "top_p": 0.9,
                "sample_count": 1,
                "start_date": "2024-01-01"
            }
        },
        {
            "name": "测试3: start_date为null字符串",
            "params": {
                "file_path": "stock_600519_live",
                "lookback": 100,
                "pred_len": 30,
                "temperature": 1.0,
                "top_p": 0.9,
                "sample_count": 1,
                "start_date": "null"
            }
        },
        {
            "name": "测试4: start_date为undefined字符串",
            "params": {
                "file_path": "stock_600519_live",
                "lookback": 100,
                "pred_len": 30,
                "temperature": 1.0,
                "top_p": 0.9,
                "sample_count": 1,
                "start_date": "undefined"
            }
        },
        {
            "name": "测试5: start_date为空字符串",
            "params": {
                "file_path": "stock_600519_live",
                "lookback": 100,
                "pred_len": 30,
                "temperature": 1.0,
                "top_p": 0.9,
                "sample_count": 1,
                "start_date": ""
            }
        },
        {
            "name": "测试6: 不带start_date参数",
            "params": {
                "file_path": "stock_600519_live",
                "lookback": 100,
                "pred_len": 30,
                "temperature": 1.0,
                "top_p": 0.9,
                "sample_count": 1
            }
        },
        {
            "name": "测试7: 模拟前端currentDataFile格式",
            "params": {
                "file_path": "stock_600519_live",
                "lookback": 400,
                "pred_len": 120,
                "temperature": 1.3,
                "top_p": 0.98,
                "sample_count": 2,
                "start_date": None
            }
        }
    ]
    
    url = "http://localhost:8080/api/predict"
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"📋 {test_case['name']}")
        print(f"{'='*60}")
        
        params = test_case['params']
        print(f"请求参数: {json.dumps(params, indent=2, ensure_ascii=False)}")
        
        try:
            # 发送预测请求
            response = requests.post(url, json=params, timeout=30)
            
            print(f"📊 响应状态码: {response.status_code}")
            print(f"📊 响应头: {dict(response.headers)}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ 预测请求成功！")
                print(f"预测类型: {result.get('prediction_type', 'N/A')}")
                print(f"消息: {result.get('message', 'N/A')}")
                print(f"预测点数: {len(result.get('prediction_results', []))}")
                print(f"实际数据点数: {len(result.get('actual_data', []))}")
            else:
                print(f"❌ 预测请求失败，状态码: {response.status_code}")
                
                # 尝试获取详细的错误信息
                try:
                    error_data = response.json()
                    print(f"错误信息: {json.dumps(error_data, indent=2, ensure_ascii=False)}")
                    
                    # 分析错误类型
                    if 'error' in error_data:
                        error_msg = error_data['error']
                        print(f"🔍 错误分析:")
                        if 'file_path' in error_msg.lower():
                            print("   - 可能原因: 文件路径无效或数据文件不存在")
                        elif 'lookback' in error_msg.lower():
                            print("   - 可能原因: lookback参数超出数据范围")
                        elif 'start_date' in error_msg.lower():
                            print("   - 可能原因: 起始日期格式无效")
                        elif 'data' in error_msg.lower():
                            print("   - 可能原因: 数据加载失败或数据量不足")
                        else:
                            print("   - 可能原因: 其他参数验证失败")
                            
                except json.JSONDecodeError:
                    print(f"原始响应内容: {response.text}")
                    
        except requests.exceptions.ConnectionError:
            print("❌ 连接失败 - 请确保后端服务器正在运行在8080端口")
        except requests.exceptions.Timeout:
            print("❌ 请求超时 - 预测可能需要更长时间")
        except Exception as e:
            print(f"❌ 请求异常: {str(e)}")
        
        print("-" * 60)

def test_specific_400_error():
    """测试特定的400错误场景"""
    
    print("\n" + "="*80)
    print("🔍 测试特定的400错误场景")
    print("="*80)
    
    # 模拟前端可能发送的有问题的参数
    problematic_params = [
        {
            "name": "无效的file_path格式",
            "params": {
                "file_path": "invalid_file_path",
                "lookback": 100,
                "pred_len": 30
            }
        },
        {
            "name": "过大的lookback值",
            "params": {
                "file_path": "stock_600519_live",
                "lookback": 10000,
                "pred_len": 30
            }
        },
        {
            "name": "无效的start_date格式",
            "params": {
                "file_path": "stock_600519_live",
                "lookback": 100,
                "pred_len": 30,
                "start_date": "invalid_date"
            }
        },
        {
            "name": "缺少必需参数",
            "params": {
                "file_path": "stock_600519_live"
                # 缺少lookback和pred_len
            }
        }
    ]
    
    url = "http://localhost:8080/api/predict"
    
    for test_case in problematic_params:
        print(f"\n📋 {test_case['name']}")
        print(f"请求参数: {json.dumps(test_case['params'], indent=2, ensure_ascii=False)}")
        
        try:
            response = requests.post(url, json=test_case['params'], timeout=10)
            print(f"状态码: {response.status_code}")
            
            if response.status_code != 200:
                try:
                    error_data = response.json()
                    print(f"错误信息: {json.dumps(error_data, indent=2, ensure_ascii=False)}")
                except:
                    print(f"原始响应: {response.text}")
            else:
                print("✅ 请求成功（这可能表明参数验证不够严格）")
                
        except Exception as e:
            print(f"❌ 请求异常: {e}")

if __name__ == "__main__":
    print("🚀 开始测试预测API 400错误...")
    
    # 测试正常和边界情况
    test_prediction_api_with_detailed_error()
    
    # 测试特定的400错误场景
    test_specific_400_error()
    
    print("\n" + "="*80)
    print("📋 测试完成总结")
    print("="*80)
    print("💡 建议:")
    print("1. 检查前端发送的预测请求参数格式")
    print("2. 查看后端API的错误日志获取详细错误信息")
    print("3. 验证数据文件是否存在且格式正确")
    print("4. 检查参数验证逻辑是否过于严格")
    print("="*80)