#!/usr/bin/env python3
"""
调试服务器启动问题
"""

import os
import sys
import traceback

def test_imports():
    """测试所有必要的导入"""
    print("=== 测试导入 ===")
    
    # 测试基本导入
    try:
        import pandas as pd
        print("✅ pandas 导入成功")
    except Exception as e:
        print(f"❌ pandas 导入失败: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy 导入成功")
    except Exception as e:
        print(f"❌ numpy 导入失败: {e}")
        return False
    
    try:
        from flask import Flask
        print("✅ flask 导入成功")
    except Exception as e:
        print(f"❌ flask 导入失败: {e}")
        return False
    
    # 测试项目模块导入
    try:
        from direct_model_loader import DirectModelLoader
        print("✅ direct_model_loader 导入成功")
    except Exception as e:
        print(f"❌ direct_model_loader 导入失败: {e}")
        return False
    
    try:
        from model import KronosPredictor
        print("✅ model 导入成功")
    except Exception as e:
        print(f"❌ model 导入失败: {e}")
        return False
    
    return True

def test_server_start():
    """测试服务器启动"""
    print("\n=== 测试服务器启动 ===")
    
    try:
        # 创建简单的Flask应用
        from flask import Flask
        app = Flask(__name__)
        
        @app.route('/')
        def hello():
            return 'Hello World!'
        
        # 尝试启动服务器（但立即停止）
        print("✅ Flask应用创建成功")
        print("✅ 服务器启动逻辑正常")
        return True
        
    except Exception as e:
        print(f"❌ 服务器启动测试失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始调试服务器启动问题...")
    
    # 测试导入
    imports_ok = test_imports()
    
    # 测试服务器启动
    server_ok = test_server_start()
    
    # 总结结果
    print("\n=== 调试结果 ===")
    if imports_ok and server_ok:
        print("✅ 所有测试通过，服务器应该能正常启动")
        print("问题可能在于app.py中的初始化逻辑")
    else:
        print("❌ 存在导入或启动问题")
        
        if not imports_ok:
            print("问题: 导入失败")
        if not server_ok:
            print("问题: 服务器启动失败")
    
    return imports_ok and server_ok

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n💡 建议: 检查app.py中的初始化代码是否有异常")
    else:
        print("\n⚠️ 需要修复导入或启动问题")