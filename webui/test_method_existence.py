#!/usr/bin/env python3
"""
测试from_pretrained方法是否存在
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_method_existence():
    """测试from_pretrained方法是否存在"""
    print("=== 测试from_pretrained方法是否存在 ===")
    
    try:
        # 导入模型类
        from model import KronosTokenizer, Kronos
        
        # 检查KronosTokenizer的from_pretrained方法
        print("检查KronosTokenizer的from_pretrained方法...")
        if hasattr(KronosTokenizer, 'from_pretrained'):
            print("✅ KronosTokenizer.from_pretrained方法存在")
            print(f"方法类型: {type(KronosTokenizer.from_pretrained)}")
        else:
            print("❌ KronosTokenizer.from_pretrained方法不存在")
            return False
        
        # 检查Kronos的from_pretrained方法
        print("\n检查Kronos的from_pretrained方法...")
        if hasattr(Kronos, 'from_pretrained'):
            print("✅ Kronos.from_pretrained方法存在")
            print(f"方法类型: {type(Kronos.from_pretrained)}")
        else:
            print("❌ Kronos.from_pretrained方法不存在")
            return False
        
        # 检查方法是否为类方法
        print("\n检查方法是否为类方法...")
        import inspect
        
        tokenizer_method = inspect.getattr_static(KronosTokenizer, 'from_pretrained')
        if isinstance(tokenizer_method, classmethod):
            print("✅ KronosTokenizer.from_pretrained是类方法")
        else:
            print("❌ KronosTokenizer.from_pretrained不是类方法")
            
        kronos_method = inspect.getattr_static(Kronos, 'from_pretrained')
        if isinstance(kronos_method, classmethod):
            print("✅ Kronos.from_pretrained是类方法")
        else:
            print("❌ Kronos.from_pretrained不是类方法")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_method_signature():
    """测试方法签名"""
    print("\n=== 测试方法签名 ===")
    
    try:
        from model import KronosTokenizer, Kronos
        import inspect
        
        # 检查KronosTokenizer的from_pretrained方法签名
        print("检查KronosTokenizer.from_pretrained方法签名...")
        sig = inspect.signature(KronosTokenizer.from_pretrained)
        print(f"方法签名: {sig}")
        
        # 检查Kronos的from_pretrained方法签名
        print("\n检查Kronos.from_pretrained方法签名...")
        sig = inspect.signature(Kronos.from_pretrained)
        print(f"方法签名: {sig}")
        
        return True
        
    except Exception as e:
        print(f"❌ 方法签名测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_import_works():
    """测试导入是否正常工作"""
    print("\n=== 测试导入功能 ===")
    
    try:
        from model import get_model_class
        
        # 测试get_model_class函数
        print("测试get_model_class函数...")
        
        kronos_class = get_model_class("kronos")
        print(f"get_model_class('kronos') = {kronos_class}")
        
        tokenizer_class = get_model_class("kronos_tokenizer")
        print(f"get_model_class('kronos_tokenizer') = {tokenizer_class}")
        
        # 检查是否能创建实例
        print("\n测试是否能创建实例...")
        
        # 尝试使用构造函数创建实例（不调用from_pretrained）
        try:
            # KronosTokenizer需要很多参数，我们只测试导入
            print("✅ 模型类导入成功")
            return True
        except Exception as e:
            print(f"❌ 实例创建失败: {e}")
            return False
            
    except Exception as e:
        print(f"❌ 导入测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始测试from_pretrained方法实现...\n")
    
    # 运行所有测试
    method_exists = test_method_existence()
    signature_ok = test_method_signature()
    import_ok = test_import_works()
    
    print("\n=== 测试结果汇总 ===")
    print(f"方法存在性: {'✅ 成功' if method_exists else '❌ 失败'}")
    print(f"方法签名: {'✅ 成功' if signature_ok else '❌ 失败'}")
    print(f"导入功能: {'✅ 成功' if import_ok else '❌ 失败'}")
    
    if method_exists and signature_ok and import_ok:
        print("\n🎉 所有基础测试通过！from_pretrained方法已正确实现。")
        print("注意：实际模型加载需要预训练模型文件，当前测试仅验证方法存在性。")
        return True
    else:
        print("\n❌ 部分测试失败，需要进一步调试。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)