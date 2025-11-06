#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复预测API数据量不足问题
"""

import requests
import json
import time

def test_data_provider_availability():
    """测试数据提供者的可用性"""
    print("\n=== 测试数据提供者可用性 ===")
    
    # 测试搜索功能
    try:
        response = requests.post(
            "http://localhost:7070/api/akshare/search-stock",
            json={"keyword": "600519"},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 股票搜索功能正常，结果数量: {data.get('count', 0)}")
            if data.get('results'):
                for result in data['results'][:3]:  # 显示前3个结果
                    print(f"   - {result.get('code', 'N/A')}: {result.get('name', 'N/A')}")
            return True
        else:
            print(f"❌ 股票搜索失败，状态码: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 股票搜索异常: {e}")
        return False

def test_stock_data_retrieval():
    """测试股票数据获取功能"""
    print("\n=== 测试股票数据获取 ===")
    
    # 测试获取股票历史数据
    try:
        response = requests.post(
            "http://localhost:7070/api/akshare/get-stock-data",
            json={
                "symbol": "600519",
                "period": "daily",
                "start_date": "20240101",
                "end_date": "20241104"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 股票数据获取成功")
            if data.get('success'):
                print(f"  数据信息: {data.get('data_info', {})}")
                if data.get('data_info') and len(data['data_info']) > 0:
                    print(f"  数据量: {len(data['data_info'])} 条记录")
                    print(f"  时间范围: {data['data_info'][0].get('date', 'N/A')} 到 {data['data_info'][-1].get('date', 'N/A')}")
            return True
        else:
            print(f"❌ 股票数据获取失败，状态码: {response.status_code}")
            try:
                error_data = response.json()
                print(f"  错误信息: {error_data.get('error', 'N/A')}")
            except:
                print(f"  响应内容: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 股票数据获取异常: {e}")
        return False

def test_download_stock_data():
    """测试下载股票数据功能"""
    print("\n=== 测试下载股票数据 ===")
    
    try:
        response = requests.post(
            "http://localhost:7070/api/akshare/download-stock-data",
            json={"symbol": "600519"},
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 股票数据下载成功")
            print(f"  文件路径: {data.get('file_path', 'N/A')}")
            print(f"  数据量: {data.get('data_count', 'N/A')} 条记录")
            return True
        else:
            print(f"❌ 股票数据下载失败，状态码: {response.status_code}")
            try:
                error_data = response.json()
                print(f"  错误信息: {error_data.get('error', 'N/A')}")
            except:
                print(f"  响应内容: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 股票数据下载异常: {e}")
        return False

def test_predict_with_downloaded_data():
    """测试使用下载的数据进行预测"""
    print("\n=== 测试使用下载的数据进行预测 ===")
    
    # 先下载数据
    download_result = test_download_stock_data()
    
    if not download_result:
        print("❌ 数据下载失败，无法进行预测测试")
        return False
    
    # 等待数据下载完成
    time.sleep(2)
    
    # 测试预测API（使用下载的数据文件）
    try:
        response = requests.post(
            "http://localhost:7070/api/predict",
            json={
                "file_path": "stock_600519_20241104.csv",  # 假设下载的文件名
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

def test_alternative_predict_parameters():
    """测试替代的预测参数"""
    print("\n=== 测试替代预测参数 ===")
    
    test_cases = [
        {
            "name": "简化参数（小数据量）",
            "params": {
                "file_path": "stock_600519_live",
                "lookback": 50,  # 最小要求
                "pred_len": 10,
                "temperature": 1.0,
                "top_p": 0.9,
                "sample_count": 1
            }
        },
        {
            "name": "极简参数",
            "params": {
                "file_path": "stock_600519_live",
                "lookback": 30,  # 低于最小要求，测试智能调整
                "pred_len": 5,
                "temperature": 1.0,
                "top_p": 0.9,
                "sample_count": 1
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n测试: {test_case['name']}")
        
        try:
            response = requests.post(
                "http://localhost:7070/api/predict",
                json=test_case['params'],
                timeout=30
            )
            
            if response.status_code == 200:
                print("✅ 请求成功")
            else:
                print(f"❌ 请求失败，状态码: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"  错误信息: {error_data.get('error', 'N/A')}")
                except:
                    print(f"  响应内容: {response.text}")
                    
        except Exception as e:
            print(f"❌ 请求异常: {e}")

def main():
    """主测试函数"""
    print("开始诊断和修复预测API数据量不足问题...")
    
    # 测试数据提供者可用性
    provider_ok = test_data_provider_availability()
    
    # 测试股票数据获取
    data_retrieval_ok = test_stock_data_retrieval()
    
    # 测试下载功能
    download_ok = test_download_stock_data()
    
    # 测试使用下载的数据进行预测
    predict_ok = test_predict_with_downloaded_data()
    
    # 测试替代参数
    test_alternative_predict_parameters()
    
    # 总结结果
    print("\n=== 诊断结果总结 ===")
    print(f"数据提供者可用性: {'✅ 正常' if provider_ok else '❌ 异常'}")
    print(f"股票数据获取: {'✅ 正常' if data_retrieval_ok else '❌ 异常'}")
    print(f"数据下载功能: {'✅ 正常' if download_ok else '❌ 异常'}")
    print(f"预测功能: {'✅ 正常' if predict_ok else '❌ 异常'}")
    
    if not predict_ok:
        print("\n⚠️ 预测API存在问题，建议解决方案:")
        print("1. 确保数据提供者正常工作")
        print("2. 使用下载的数据文件进行预测（而非实时数据）")
        print("3. 检查网络连接和akshare API可用性")
        print("4. 调整预测参数，使用更小的lookback值")
    else:
        print("\n🎉 预测API工作正常！")

if __name__ == "__main__":
    main()