#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试预测API的各种参数组合，验证修复效果
"""

import requests
import json
import sys

def test_parameter_combinations():
    """测试各种参数组合"""
    
    # API端点
    url = "http://localhost:7070/api/predict"
    
    # 测试用例 - 各种边界情况和参数组合
    test_cases = [
        # 基础正常测试
        {
            "name": "基础正常测试",
            "params": {
                "file_path": "stock_600519_live",
                "pred_len": 5,
                "lookback": 100,
                "temperature": 1.0,
                "top_p": 0.9,
                "sample_count": 1
            },
            "expected_status": 200
        },
        
        # 边界值测试
        {
            "name": "最小预测长度",
            "params": {
                "file_path": "stock_600519_live",
                "pred_len": 1,
                "lookback": 50,
                "temperature": 1.0,
                "top_p": 0.9,
                "sample_count": 1
            },
            "expected_status": 200
        },
        
        {
            "name": "最小回看期",
            "params": {
                "file_path": "stock_600519_live",
                "pred_len": 5,
                "lookback": 10,
                "temperature": 1.0,
                "top_p": 0.9,
                "sample_count": 1
            },
            "expected_status": 200
        },
        
        # 不同股票代码测试
        {
            "name": "其他股票代码",
            "params": {
                "file_path": "stock_000001_live",
                "pred_len": 5,
                "lookback": 100,
                "temperature": 1.0,
                "top_p": 0.9,
                "sample_count": 1
            },
            "expected_status": 200
        },
        
        # 错误参数测试（应该返回400）
        {
            "name": "无效文件路径",
            "params": {
                "file_path": "invalid_stock_data",
                "pred_len": 5,
                "lookback": 100,
                "temperature": 1.0,
                "top_p": 0.9,
                "sample_count": 1
            },
            "expected_status": 400
        },
        
        {
            "name": "预测长度过大",
            "params": {
                "file_path": "stock_600519_live",
                "pred_len": 1000,
                "lookback": 100,
                "temperature": 1.0,
                "top_p": 0.9,
                "sample_count": 1
            },
            "expected_status": 400
        },
        
        {
            "name": "回看期过大",
            "params": {
                "file_path": "stock_600519_live",
                "pred_len": 5,
                "lookback": 10000,
                "temperature": 1.0,
                "top_p": 0.9,
                "sample_count": 1
            },
            "expected_status": 400
        },
        
        {
            "name": "缺少必需参数",
            "params": {
                "file_path": "stock_600519_live",
                "pred_len": 5,
                "lookback": 100
                # 缺少 temperature, top_p, sample_count
            },
            "expected_status": 400
        }
    ]
    
    print("=== 预测API参数组合测试 ===")
    print("测试修复后的API对各种参数组合的处理能力\n")
    
    passed_tests = 0
    failed_tests = 0
    
    for test_case in test_cases:
        print(f"测试: {test_case['name']}")
        print(f"参数: {json.dumps(test_case['params'], indent=2, ensure_ascii=False)}")
        print(f"期望状态码: {test_case['expected_status']}")
        
        try:
            # 发送POST请求
            response = requests.post(url, json=test_case['params'], timeout=30)
            
            actual_status = response.status_code
            print(f"实际状态码: {actual_status}")
            
            # 检查状态码是否符合预期
            if actual_status == test_case['expected_status']:
                print("✓ 状态码符合预期")
                
                # 对于成功响应，进一步检查响应内容
                if actual_status == 200:
                    result = response.json()
                    if result.get('success'):
                        print("✓ 预测成功")
                        
                        # 检查关键字段是否存在
                        required_fields = ['prediction_results', 'actual_data', 'chart', 'message']
                        missing_fields = [field for field in required_fields if field not in result]
                        
                        if not missing_fields:
                            print("✓ 响应包含所有必需字段")
                            
                            # 检查预测结果长度
                            pred_len = test_case['params']['pred_len']
                            actual_pred_len = len(result.get('prediction_results', []))
                            
                            if actual_pred_len == pred_len:
                                print(f"✓ 预测结果长度正确: {actual_pred_len}")
                            else:
                                print(f"⚠ 预测结果长度不匹配: 期望{pred_len}, 实际{actual_pred_len}")
                                
                            # 检查实际数据是否存在
                            actual_data_len = len(result.get('actual_data', []))
                            if actual_data_len > 0:
                                print(f"✓ 实际数据存在: {actual_data_len} 个数据点")
                            else:
                                print("⚠ 实际数据为空")
                                
                        else:
                            print(f"✗ 缺少必需字段: {missing_fields}")
                            failed_tests += 1
                            
                    else:
                        print(f"✗ 预测失败: {result.get('error', '未知错误')}")
                        failed_tests += 1
                        
                # 对于400错误，检查错误信息
                elif actual_status == 400:
                    result = response.json()
                    if 'error' in result:
                        print(f"✓ 错误信息: {result['error']}")
                    else:
                        print("⚠ 400响应缺少错误信息")
                        
                passed_tests += 1
                
            else:
                print(f"✗ 状态码不符合预期")
                failed_tests += 1
                
                # 打印错误详情
                if actual_status != 200:
                    try:
                        result = response.json()
                        print(f"错误详情: {result}")
                    except:
                        print(f"响应内容: {response.text}")
                        
        except requests.exceptions.RequestException as e:
            print(f"✗ 请求异常: {e}")
            failed_tests += 1
        except Exception as e:
            print(f"✗ 未知异常: {e}")
            failed_tests += 1
            
        print("-" * 50)
    
    # 测试总结
    print("\n=== 测试总结 ===")
    print(f"总测试数: {len(test_cases)}")
    print(f"通过测试: {passed_tests}")
    print(f"失败测试: {failed_tests}")
    print(f"通过率: {passed_tests/len(test_cases)*100:.1f}%")
    
    if failed_tests == 0:
        print("🎉 所有测试通过！预测API修复成功！")
        return True
    else:
        print("⚠ 部分测试失败，需要进一步检查")
        return False

if __name__ == "__main__":
    success = test_parameter_combinations()
    sys.exit(0 if success else 1)