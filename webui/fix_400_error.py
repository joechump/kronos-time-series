#!/usr/bin/env python3
"""
修复预测API 400错误问题
通过模拟前端请求，验证修复方案
"""

import requests
import json
import time

def test_frontend_request_simulation():
    """模拟前端请求，验证修复方案"""
    
    base_url = "http://localhost:8080"
    
    # 模拟前端请求的完整参数
    frontend_params = {
        "file_path": "stock_600519_live",  # 前端设置的currentDataFile
        "lookback": 100,
        "pred_len": 30,
        "start_date": None,
        "temperature": 1.3,
        "top_p": 0.98,
        "sample_count": 2
    }
    
    print("🔍 模拟前端预测请求...")
    print("=" * 80)
    print(f"📤 请求参数: {json.dumps(frontend_params, indent=2, ensure_ascii=False)}")
    
    try:
        # 发送POST请求
        response = requests.post(
            f"{base_url}/api/predict",
            json=frontend_params,
            timeout=60  # 60秒超时
        )
        
        print(f"📥 响应状态: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 预测成功!")
            print(f"📊 消息: {result.get('message', '无消息')}")
            print(f"🔢 预测点数: {result.get('prediction_count', 0)}")
            print(f"📈 实际数据点数: {result.get('actual_data_count', 0)}")
            
            # 显示预测结果摘要
            if 'prediction' in result:
                pred_data = result['prediction']
                if isinstance(pred_data, list) and len(pred_data) > 0:
                    print(f"📊 预测结果摘要: 共{len(pred_data)}个预测点")
                    print(f"   第一个预测点: {pred_data[0]}")
                    print(f"   最后一个预测点: {pred_data[-1]}")
            
            return True
            
        elif response.status_code == 400:
            error_data = response.json()
            print(f"❌ 400错误: {error_data.get('error', '未知错误')}")
            print(f"🔍 错误详情: {error_data}")
            
            # 分析错误原因
            error_msg = error_data.get('error', '')
            if '文件路径不能为空' in error_msg:
                print("💡 问题分析: 文件路径参数为空")
                print("🛠️ 修复建议: 检查前端currentDataFile变量是否正确设置")
            elif 'No such file' in error_msg:
                print("💡 问题分析: 文件不存在")
                print("🛠️ 修复建议: 检查数据文件是否已正确下载")
            else:
                print("💡 问题分析: 其他参数验证错误")
                print("🛠️ 修复建议: 检查所有必填参数")
            
            return False
            
        else:
            print(f"⚠️ 其他错误: {response.status_code}")
            print(f"📝 响应内容: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("⏰ 请求超时: 预测过程可能耗时较长")
        return False
    except requests.exceptions.ConnectionError:
        print("🔌 连接错误: 请检查后端服务是否正常运行")
        return False
    except Exception as e:
        print(f"💥 请求异常: {e}")
        return False

def check_data_file_exists():
    """检查数据文件是否存在"""
    
    print("\n🔍 检查数据文件状态...")
    print("-" * 60)
    
    base_url = "http://localhost:8080"
    
    # 检查股票数据文件
    try:
        response = requests.post(
            f"{base_url}/api/akshare/check-stock-data",
            json={"symbol": "600519"},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 股票数据检查成功")
            print(f"📊 数据信息: {data}")
            return True
        else:
            print(f"❌ 股票数据检查失败: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"💥 数据检查异常: {e}")
        return False

def verify_frontend_parameters():
    """验证前端参数设置"""
    
    print("\n🔍 验证前端参数设置...")
    print("-" * 60)
    
    # 正确的参数格式
    correct_params = {
        "file_path": "stock_600519_live",  # 不带扩展名
        "lookback": 100,
        "pred_len": 30,
        "start_date": None,
        "temperature": 1.3,
        "top_p": 0.98,
        "sample_count": 2
    }
    
    print("✅ 正确的参数格式:")
    print(json.dumps(correct_params, indent=2, ensure_ascii=False))
    
    # 错误的参数格式示例
    wrong_params = [
        {"file_path": "stock_600519_live.csv"},  # 带扩展名
        {"file_path": "c:\\kron\\data\\stock_600519_live.csv"},  # 绝对路径
        {"data_file": "stock_600519_live"},  # 错误的参数名
        {"file_path": ""}  # 空路径
    ]
    
    print("\n❌ 错误的参数格式示例:")
    for i, params in enumerate(wrong_params, 1):
        print(f"   {i}. {params}")

def main():
    """主函数"""
    
    print("🔍 Kronos 预测API 400错误修复验证")
    print("=" * 80)
    
    # 验证前端参数设置
    verify_frontend_parameters()
    
    # 检查数据文件状态
    check_data_file_exists()
    
    # 模拟前端请求
    success = test_frontend_request_simulation()
    
    print("\n" + "=" * 80)
    print("📋 修复验证结果:")
    print("-" * 60)
    
    if success:
        print("✅ 修复成功! 预测API现在可以正常工作")
        print("💡 前端参数设置正确，后端API响应正常")
    else:
        print("❌ 修复失败，需要进一步排查")
        print("🔍 可能的问题:")
        print("   1. 数据文件未正确下载")
        print("   2. 后端服务异常")
        print("   3. 参数格式仍有问题")
    
    print("\n🛠️ 下一步操作:")
    print("   1. 在前端页面重新加载股票数据")
    print("   2. 确保模型状态显示为'已加载完成'")
    print("   3. 点击预测按钮进行测试")

if __name__ == "__main__":
    main()