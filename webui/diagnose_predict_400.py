#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断预测API 400错误问题
"""

import requests
import json
import sys
import os

def test_model_status():
    """测试模型状态API"""
    print("=== 测试模型状态API ===")
    
    url = "http://localhost:7070/api/model-status"
    
    try:
        response = requests.get(url, timeout=10)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 模型状态API正常")
            print(f"模型可用性: {result.get('model_available', 'N/A')}")
            print(f"直接模型加载: {result.get('direct_model_loaded', 'N/A')}")
            print(f"模型名称: {result.get('model_name', 'N/A')}")
            return True
        else:
            print(f"❌ 模型状态API返回错误: {response.status_code}")
            print(f"响应内容: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务器")
        return False
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        return False

def test_load_model():
    """测试模型加载API"""
    print("\n=== 测试模型加载API ===")
    
    url = "http://localhost:7070/api/load-model"
    
    # 尝试加载kronos-small模型
    data = {
        "model_name": "kronos-small"
    }
    
    try:
        response = requests.post(url, json=data, timeout=30)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 模型加载API正常")
            print(f"加载结果: {result.get('message', 'N/A')}")
            return True
        else:
            print(f"❌ 模型加载API返回错误: {response.status_code}")
            try:
                error_data = response.json()
                print(f"错误信息: {error_data}")
            except:
                print(f"响应内容: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务器")
        return False
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        return False

def test_predict_api_with_detailed_error():
    """测试预测API并获取详细错误信息"""
    print("\n=== 测试预测API（模拟前端请求） ===")
    
    # 模拟前端发送的预测请求参数
    prediction_params = {
        "file_path": "stock_600523_live",
        "lookback": 400,
        "pred_len": 120,
        "temperature": 1.3,
        "top_p": 1,
        "sample_count": 1,
        "trading_mode": "calendar",
        "start_date": ""
    }
    
    url = "http://localhost:7070/api/predict"
    
    print(f"请求参数: {json.dumps(prediction_params, indent=2, ensure_ascii=False)}")
    
    try:
        response = requests.post(url, json=prediction_params, timeout=60)
        print(f"状态码: {response.status_code}")
        print(f"响应头: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 预测API请求成功！")
            print(f"预测类型: {result.get('prediction_type', 'N/A')}")
            print(f"消息: {result.get('message', 'N/A')}")
            return True
        else:
            print(f"❌ 预测API返回错误: {response.status_code}")
            
            # 尝试获取详细的错误信息
            try:
                error_data = response.json()
                print(f"错误信息: {json.dumps(error_data, indent=2, ensure_ascii=False)}")
                
                # 分析错误类型
                error_msg = error_data.get('error', '')
                if '模型' in error_msg and '不可用' in error_msg:
                    print("🔍 错误分析: 模型未正确加载")
                elif '数据' in error_msg and '不足' in error_msg:
                    print("🔍 错误分析: 数据量不足")
                elif '文件路径' in error_msg:
                    print("🔍 错误分析: 文件路径问题")
                elif 'Akshare' in error_msg:
                    print("🔍 错误分析: 数据提供者问题")
                else:
                    print("🔍 错误分析: 其他问题")
                    
            except:
                print(f"原始响应内容: {response.text}")
            
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务器")
        return False
    except requests.exceptions.Timeout:
        print("❌ 请求超时")
        return False
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        return False

def check_direct_model_loader():
    """检查直接模型加载器的状态"""
    print("\n=== 检查直接模型加载器状态 ===")
    
    try:
        from direct_model_loader import DirectModelLoader
        
        loader = DirectModelLoader()
        print("✅ DirectModelLoader导入成功")
        
        # 检查可用模型
        available_models = loader.available_models
        print(f"可用模型: {list(available_models.keys())}")
        
        # 检查每个模型的详细信息
        for model_key, model_info in available_models.items():
            print(f"\n模型 {model_key}:")
            print(f"  状态: {model_info.get('status', 'unknown')}")
            print(f"  路径: {model_info.get('local_path', 'unknown')}")
            
            # 检查必要的文件是否存在
            if model_info.get('local_path'):
                config_file = os.path.join(model_info['local_path'], 'config.json')
                model_file = os.path.join(model_info['local_path'], 'model.safetensors')
                
                print(f"  配置文件存在: {os.path.exists(config_file)}")
                print(f"  模型文件存在: {os.path.exists(model_file)}")
        
        # 检查已加载的模型
        loaded_model = loader.get_loaded_model()
        print(f"\n已加载模型: {loaded_model is not None}")
        
        if loaded_model:
            print("✅ 模型已正确加载")
            return True
        else:
            print("❌ 模型未加载")
            return False
            
    except Exception as e:
        print(f"❌ DirectModelLoader检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主诊断函数"""
    print("🚀 开始诊断预测API 400错误问题")
    print("=" * 60)
    
    # 1. 测试模型状态API
    model_status_ok = test_model_status()
    
    # 2. 如果模型状态有问题，尝试加载模型
    if not model_status_ok:
        print("\n⚠️ 模型状态异常，尝试加载模型...")
        load_model_ok = test_load_model()
        
        if load_model_ok:
            # 重新测试模型状态
            print("\n重新测试模型状态...")
            model_status_ok = test_model_status()
    
    # 3. 检查直接模型加载器状态
    direct_loader_ok = check_direct_model_loader()
    
    # 4. 测试预测API
    predict_ok = test_predict_api_with_detailed_error()
    
    # 总结诊断结果
    print("\n" + "=" * 60)
    print("📊 诊断结果总结:")
    print(f"   模型状态API: {'✅ 正常' if model_status_ok else '❌ 异常'}")
    print(f"   直接模型加载器: {'✅ 正常' if direct_loader_ok else '❌ 异常'}")
    print(f"   预测API: {'✅ 正常' if predict_ok else '❌ 异常'}")
    
    if predict_ok:
        print("\n🎉 预测API 400错误已解决！")
    else:
        print("\n⚠️ 预测API仍存在问题，需要进一步调试")
        
        # 提供修复建议
        if not model_status_ok:
            print("\n💡 修复建议:")
            print("1. 检查模型文件是否完整")
            print("2. 尝试手动加载模型: POST http://localhost:7070/api/load-model")
            print("3. 检查模型目录权限")
        elif not direct_loader_ok:
            print("\n💡 修复建议:")
            print("1. 检查direct_model_loader.py是否正确导入")
            print("2. 检查模型文件路径是否正确")
            print("3. 重启Web服务器")
    
    print("=" * 60)

if __name__ == "__main__":
    main()