#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细测试模型加载功能
"""

import os
import sys
import traceback

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=== 详细模型加载测试 ===")

# 测试1: 基本导入
try:
    print("\n1. 测试基本导入...")
    from model import KronosTokenizer, Kronos
    print("✓ 基本导入成功")
except Exception as e:
    print(f"✗ 基本导入失败: {e}")
    traceback.print_exc()
    sys.exit(1)

# 测试2: 检查from_pretrained方法是否存在
print("\n2. 检查from_pretrained方法...")
if hasattr(KronosTokenizer, 'from_pretrained'):
    print("✓ KronosTokenizer.from_pretrained方法存在")
else:
    print("✗ KronosTokenizer.from_pretrained方法不存在")
    sys.exit(1)

if hasattr(Kronos, 'from_pretrained'):
    print("✓ Kronos.from_pretrained方法存在")
else:
    print("✗ Kronos.from_pretrained方法不存在")
    sys.exit(1)

# 测试3: 尝试加载tokenizer
print("\n3. 尝试加载tokenizer...")
try:
    print("   使用模型ID: NeoQuasar/Kronos-Tokenizer-2k")
    tokenizer = KronosTokenizer.from_pretrained('NeoQuasar/Kronos-Tokenizer-2k')
    print("   ✓ Tokenizer加载成功")
    print(f"   Tokenizer类型: {type(tokenizer)}")
    
    # 检查tokenizer的基本属性
    if hasattr(tokenizer, 'd_in'):
        print(f"   d_in: {tokenizer.d_in}")
    if hasattr(tokenizer, 'd_model'):
        print(f"   d_model: {tokenizer.d_model}")
        
except Exception as e:
    print(f"   ✗ Tokenizer加载失败: {e}")
    traceback.print_exc()
    sys.exit(1)

# 测试4: 尝试加载模型
print("\n4. 尝试加载模型...")
try:
    print("   使用模型ID: NeoQuasar/Kronos-small")
    model = Kronos.from_pretrained('NeoQuasar/Kronos-small')
    print("   ✓ 模型加载成功")
    print(f"   模型类型: {type(model)}")
    
    # 检查模型的基本属性
    if hasattr(model, 's1_bits'):
        print(f"   s1_bits: {model.s1_bits}")
    if hasattr(model, 's2_bits'):
        print(f"   s2_bits: {model.s2_bits}")
        
except Exception as e:
    print(f"   ✗ 模型加载失败: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n=== 所有测试通过！ ===")
print("模型加载功能正常工作。")