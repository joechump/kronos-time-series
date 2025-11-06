#!/usr/bin/env python3
"""
测试修复后的模型加载逻辑
"""

import sys
import os
sys.path.append('..')

from direct_model_loader import DirectModelLoader

def test_direct_model_loader():
    """测试直接模型加载器"""
    print("=== 测试直接模型加载器 ===")
    
    # 创建模型加载器
    loader = DirectModelLoader()
    
    # 检查可用模型
    print(f"可用模型: {list(loader.available_models.keys())}")
    
    # 尝试加载kronos-small模型
    if 'kronos-small' in loader.available_models:
        print("\n=== 测试加载kronos-small模型 ===")
        success, message = loader.load_model('kronos-small')
        print(f"加载结果: {success}")
        print(f"消息: {message}")
        
        if success:
            # 获取加载的模型
            model_data = loader.get_loaded_model('kronos-small')
            if model_data:
                print(f"模型类型: {type(model_data['model'])}")
                print(f"Tokenizer类型: {type(model_data['tokenizer'])}")
                print("模型加载成功！")
                return True
    
    return False

def test_auto_load():
    """测试自动加载最优模型"""
    print("\n=== 测试自动加载最优模型 ===")
    
    loader = DirectModelLoader()
    success, message, model_key = loader.auto_load_best_model()
    
    print(f"自动加载结果: {success}")
    print(f"消息: {message}")
    print(f"模型键: {model_key}")
    
    return success

if __name__ == "__main__":
    print("开始测试修复后的模型加载逻辑...")
    
    try:
        # 测试直接模型加载
        direct_success = test_direct_model_loader()
        
        # 测试自动加载
        auto_success = test_auto_load()
        
        if direct_success and auto_success:
            print("\n✅ 所有测试通过！模型加载逻辑修复成功！")
        else:
            print("\n❌ 部分测试失败，需要进一步调试")
            
    except Exception as e:
        import traceback
        print(f"\n❌ 测试过程中出现异常: {str(e)}")
        print(f"详细错误信息:\n{traceback.format_exc()}")