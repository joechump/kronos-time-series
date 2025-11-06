#!/usr/bin/env python3
"""
测试服务器启动问题的脚本
"""

import sys
import os

# 添加项目根目录到系统路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.append(os.path.join(project_root, 'model'))

print("=== 测试服务器启动 ===")
print(f"项目根目录: {project_root}")
print(f"系统路径: {sys.path}")

# 测试导入
print("\n=== 测试导入 ===")
try:
    import pandas as np
    print("✓ pandas 导入成功")
except ImportError as e:
    print(f"✗ pandas 导入失败: {e}")

try:
    import numpy as np
    print("✓ numpy 导入成功")
except ImportError as e:
    print(f"✗ numpy 导入失败: {e}")

try:
    from flask import Flask
    print("✓ flask 导入成功")
except ImportError as e:
    print(f"✗ flask 导入失败: {e}")

try:
    from direct_model_loader import DirectModelLoader
    print("✓ direct_model_loader 导入成功")
except ImportError as e:
    print(f"✗ direct_model_loader 导入失败: {e}")

try:
    import model
    print("✓ model 模块导入成功")
    print(f"model 模块路径: {model.__file__}")
except ImportError as e:
    print(f"✗ model 模块导入失败: {e}")

# 测试创建Flask应用
print("\n=== 测试Flask应用创建 ===")
try:
    app = Flask(__name__)
    print("✓ Flask应用创建成功")
    
    # 测试路由
    @app.route('/test')
    def test():
        return 'Hello, World!'
    
    print("✓ 测试路由添加成功")
    
    # 测试服务器启动
    print("\n=== 测试服务器启动 ===")
    print("启动服务器在端口 8081...")
    
    # 使用threaded=True和debug=False避免立即退出
    app.run(host='0.0.0.0', port=8081, debug=False, threaded=True, use_reloader=False)
    
    print("服务器已启动")
    
except Exception as e:
    print(f"✗ Flask应用创建或启动失败: {e}")
    import traceback
    traceback.print_exc()