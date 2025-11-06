#!/usr/bin/env python3
"""
测试修复后的模型加载功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import KronosTokenizer, Kronos

def test_kronos_tokenizer_loading():
    """测试KronosTokenizer的from_pretrained方法"""
    print("=== 测试KronosTokenizer加载 ===")
    
    try:
        # 尝试使用from_pretrained方法创建实例
        print("尝试加载KronosTokenizer...")
        
        # 由于我们没有实际的预训练模型，这里测试方法是否存在
        tokenizer = KronosTokenizer.from_pretrained("test-model")
        print("✅ KronosTokenizer.from_pretrained方法调用成功")
        print(f"Tokenizer类型: {type(tokenizer)}")
        
        # 检查tokenizer的基本属性
        if hasattr(tokenizer, 'd_in'):
            print(f"✅ Tokenizer具有d_in属性: {tokenizer.d_in}")
        else:
            print("❌ Tokenizer缺少d_in属性")
            
        return True
        
    except Exception as e:
        print(f"❌ KronosTokenizer加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_kronos_model_loading():
    """测试Kronos模型的from_pretrained方法"""
    print("\n=== 测试Kronos模型加载 ===")
    
    try:
        # 尝试使用from_pretrained方法创建实例
        print("尝试加载Kronos模型...")
        
        # 由于我们没有实际的预训练模型，这里测试方法是否存在
        model = Kronos.from_pretrained("test-model")
        print("✅ Kronos.from_pretrained方法调用成功")
        print(f"模型类型: {type(model)}")
        
        # 检查模型的基本属性
        if hasattr(model, 's1_bits'):
            print(f"✅ 模型具有s1_bits属性: {model.s1_bits}")
        else:
            print("❌ 模型缺少s1_bits属性")
            
        return True
        
    except Exception as e:
        print(f"❌ Kronos模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_import():
    """测试模型库导入"""
    print("=== 测试模型库导入 ===")
    
    try:
        # 测试是否能成功导入模型库
        from model import get_model_class
        print("✅ 模型库导入成功")
        
        # 测试get_model_class函数
        model_class = get_model_class("kronos")
        print(f"✅ get_model_class('kronos')返回: {model_class}")
        
        tokenizer_class = get_model_class("kronos_tokenizer")
        print(f"✅ get_model_class('kronos_tokenizer')返回: {tokenizer_class}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型库导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始测试修复后的模型加载功能...\n")
    
    # 运行所有测试
    import_success = test_model_import()
    tokenizer_success = test_kronos_tokenizer_loading()
    model_success = test_kronos_model_loading()
    
    print("\n=== 测试结果汇总 ===")
    print(f"模型库导入: {'✅ 成功' if import_success else '❌ 失败'}")
    print(f"Tokenizer加载: {'✅ 成功' if tokenizer_success else '❌ 失败'}")
    print(f"模型加载: {'✅ 成功' if model_success else '❌ 失败'}")
    
    if import_success and tokenizer_success and model_success:
        print("\n🎉 所有测试通过！模型加载功能已修复。")
        return True
    else:
        print("\n❌ 部分测试失败，需要进一步调试。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)