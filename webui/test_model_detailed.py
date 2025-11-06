#!/usr/bin/env python3
"""
详细测试模型加载问题的脚本
"""

import os
import sys
import traceback

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_model_import_detailed():
    """详细测试模型导入"""
    print("=== 详细测试模型导入 ===")
    
    try:
        print("1. 尝试导入Kronos模型库...")
        from model import Kronos, KronosTokenizer, KronosPredictor
        print("✅ Kronos模型库导入成功")
        
        print("2. 尝试创建KronosTokenizer实例...")
        try:
            tokenizer = KronosTokenizer.from_pretrained('NeoQuasar/Kronos-Tokenizer-2k')
            print("✅ KronosTokenizer创建成功")
            return True
        except Exception as e:
            print(f"❌ KronosTokenizer创建失败: {e}")
            print("详细错误信息:")
            traceback.print_exc()
            return False
            
    except ImportError as e:
        print(f"❌ Kronos模型库导入失败: {e}")
        print("详细错误信息:")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"❌ 其他导入错误: {e}")
        print("详细错误信息:")
        traceback.print_exc()
        return False

def test_direct_loader_simple():
    """简单测试直接模型加载器"""
    print("\n=== 简单测试直接模型加载器 ===")
    
    try:
        print("1. 尝试导入DirectModelLoader...")
        from direct_model_loader import DirectModelLoader
        print("✅ DirectModelLoader导入成功")
        
        print("2. 尝试初始化DirectModelLoader...")
        loader = DirectModelLoader()
        print("✅ DirectModelLoader初始化成功")
        
        print(f"3. 可用模型: {list(loader.available_models.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ DirectModelLoader测试失败: {e}")
        print("详细错误信息:")
        traceback.print_exc()
        return False

def test_auto_loader_simple():
    """简单测试自动模型加载器"""
    print("\n=== 简单测试自动模型加载器 ===")
    
    try:
        print("1. 尝试导入AutoModelLoader...")
        from auto_model_loader import AutoModelLoader
        print("✅ AutoModelLoader导入成功")
        
        print("2. 尝试初始化AutoModelLoader...")
        loader = AutoModelLoader()
        print("✅ AutoModelLoader初始化成功")
        
        return True
        
    except Exception as e:
        print(f"❌ AutoModelLoader测试失败: {e}")
        print("详细错误信息:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始详细测试模型加载问题...")
    
    # 测试模型导入
    model_import_ok = test_model_import_detailed()
    
    # 测试直接模型加载器
    direct_loader_ok = test_direct_loader_simple()
    
    # 测试自动模型加载器
    auto_loader_ok = test_auto_loader_simple()
    
    print("\n=== 测试总结 ===")
    print(f"模型导入: {'✅ 成功' if model_import_ok else '❌ 失败'}")
    print(f"直接模型加载器: {'✅ 成功' if direct_loader_ok else '❌ 失败'}")
    print(f"自动模型加载器: {'✅ 成功' if auto_loader_ok else '❌ 失败'}")
    
    if model_import_ok and direct_loader_ok and auto_loader_ok:
        print("\n🎉 所有测试通过！模型加载系统正常")
    else:
        print("\n⚠️ 部分测试失败，需要检查模型加载问题")