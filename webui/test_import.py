#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试模型导入功能
"""

import os
import sys

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("测试模型导入...")

try:
    print("1. 尝试导入KronosTokenizer...")
    from model import KronosTokenizer
    print("✓ KronosTokenizer导入成功")
    
    print("2. 尝试导入Kronos...")
    from model import Kronos
    print("✓ Kronos导入成功")
    
    print("3. 尝试导入KronosPredictor...")
    from model import KronosPredictor
    print("✓ KronosPredictor导入成功")
    
    print("4. 测试from_pretrained方法...")
    try:
        # 测试KronosTokenizer的from_pretrained
        print("  测试KronosTokenizer.from_pretrained...")
        tokenizer = KronosTokenizer.from_pretrained('NeoQuasar/Kronos-Tokenizer-2k')
        print("  ✓ KronosTokenizer.from_pretrained成功")
    except Exception as e:
        print(f"  ✗ KronosTokenizer.from_pretrained失败: {e}")
    
    try:
        # 测试Kronos的from_pretrained
        print("  测试Kronos.from_pretrained...")
        model = Kronos.from_pretrained('NeoQuasar/Kronos-small')
        print("  ✓ Kronos.from_pretrained成功")
    except Exception as e:
        print(f"  ✗ Kronos.from_pretrained失败: {e}")
    
    print("\n所有导入测试完成！")
    
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    import traceback
    traceback.print_exc()

except Exception as e:
    print(f"✗ 发生错误: {e}")
    import traceback
    traceback.print_exc()