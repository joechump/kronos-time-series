#!/usr/bin/env python3
"""
测试模型加载问题的脚本
"""

import os
import sys

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_model_import():
    """测试模型导入"""
    print("=== 测试模型导入 ===")
    try:
        from model import Kronos, KronosTokenizer, KronosPredictor
        print("✅ 模型库导入成功")
        
        # 测试tokenizer创建
        try:
            tokenizer = KronosTokenizer.from_pretrained('NeoQuasar/Kronos-Tokenizer-2k')
            print("✅ Tokenizer创建成功")
            return True
        except Exception as e:
            print(f"❌ Tokenizer创建失败: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ 模型库导入失败: {e}")
        return False

def test_direct_model_loader():
    """测试直接模型加载器"""
    print("\n=== 测试直接模型加载器 ===")
    try:
        from direct_model_loader import DirectModelLoader
        print("✅ 直接模型加载器导入成功")
        
        loader = DirectModelLoader()
        print("✅ 直接模型加载器初始化成功")
        
        # 检查可用模型
        print(f"可用模型: {list(loader.available_models.keys())}")
        
        # 尝试加载kronos-small模型
        if 'kronos-small' in loader.available_models:
            print("尝试加载kronos-small模型...")
            success, message = loader.load_model('kronos-small')
            print(f"加载结果: {success}, {message}")
            return success
        else:
            print("❌ kronos-small模型不可用")
            return False
            
    except Exception as e:
        print(f"❌ 直接模型加载器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_auto_model_loader():
    """测试自动模型加载器"""
    print("\n=== 测试自动模型加载器 ===")
    try:
        from auto_model_loader import AutoModelLoader
        print("✅ 自动模型加载器导入成功")
        
        loader = AutoModelLoader()
        print("✅ 自动模型加载器初始化成功")
        
        # 获取系统报告
        report = loader.get_system_report()
        print(f"系统报告: {report}")
        
        return True
        
    except Exception as e:
        print(f"❌ 自动模型加载器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始测试模型加载问题...")
    
    # 测试模型导入
    model_import_ok = test_model_import()
    
    # 测试直接模型加载器
    direct_loader_ok = test_direct_model_loader()
    
    # 测试自动模型加载器
    auto_loader_ok = test_auto_model_loader()
    
    print("\n=== 测试总结 ===")
    print(f"模型导入: {'✅ 成功' if model_import_ok else '❌ 失败'}")
    print(f"直接模型加载器: {'✅ 成功' if direct_loader_ok else '❌ 失败'}")
    print(f"自动模型加载器: {'✅ 成功' if auto_loader_ok else '❌ 失败'}")
    
    if model_import_ok and direct_loader_ok and auto_loader_ok:
        print("\n🎉 所有测试通过！模型加载系统正常")
    else:
        print("\n⚠️ 部分测试失败，需要检查模型加载问题")