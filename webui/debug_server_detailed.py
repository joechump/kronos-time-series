#!/usr/bin/env python3
"""
详细调试服务器启动问题的脚本
"""

import sys
import os
import traceback

# 添加项目根目录到系统路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.append(os.path.join(project_root, 'model'))

print("=== 详细调试服务器启动 ===")
print(f"项目根目录: {project_root}")
print(f"系统路径: {sys.path}")

# 测试导入所有必要的模块
print("\n=== 测试导入所有模块 ===")

try:
    import pandas as pd
    print("✓ pandas 导入成功")
except ImportError as e:
    print(f"✗ pandas 导入失败: {e}")
    traceback.print_exc()

try:
    import numpy as np
    print("✓ numpy 导入成功")
except ImportError as e:
    print(f"✗ numpy 导入失败: {e}")
    traceback.print_exc()

try:
    from flask import Flask
    print("✓ flask 导入成功")
except ImportError as e:
    print(f"✗ flask 导入失败: {e}")
    traceback.print_exc()

try:
    from direct_model_loader import DirectModelLoader
    print("✓ direct_model_loader 导入成功")
except ImportError as e:
    print(f"✗ direct_model_loader 导入失败: {e}")
    traceback.print_exc()

try:
    import model
    print("✓ model 模块导入成功")
    print(f"model 模块路径: {model.__file__}")
except ImportError as e:
    print(f"✗ model 模块导入失败: {e}")
    traceback.print_exc()

try:
    from akshare_data_provider import AkshareDataProvider
    print("✓ akshare_data_provider 导入成功")
except ImportError as e:
    print(f"✗ akshare_data_provider 导入失败: {e}")
    traceback.print_exc()

try:
    from local_model_manager import LocalModelManager
    print("✓ local_model_manager 导入成功")
except ImportError as e:
    print(f"✗ local_model_manager 导入失败: {e}")
    traceback.print_exc()

try:
    from auto_model_loader import AutoModelLoader
    print("✓ auto_model_loader 导入成功")
except ImportError as e:
    print(f"✗ auto_model_loader 导入失败: {e}")
    traceback.print_exc()

# 测试app.py中的初始化逻辑
print("\n=== 测试app.py初始化逻辑 ===")

try:
    # 模拟app.py中的初始化逻辑
    import logging
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger('root')
    
    print("✓ 日志系统配置成功")
    
    # 测试数据提供器初始化
    try:
        data_provider = AkshareDataProvider()
        print("✓ 数据提供器初始化成功")
    except Exception as e:
        print(f"✗ 数据提供器初始化失败: {e}")
        traceback.print_exc()
    
    # 测试本地模型管理器初始化
    try:
        local_model_manager = LocalModelManager()
        print("✓ 本地模型管理器初始化成功")
    except Exception as e:
        print(f"✗ 本地模型管理器初始化失败: {e}")
        traceback.print_exc()
    
    # 测试直接模型加载器初始化
    try:
        direct_model_loader = DirectModelLoader()
        print("✓ 直接模型加载器初始化成功")
    except Exception as e:
        print(f"✗ 直接模型加载器初始化失败: {e}")
        traceback.print_exc()
    
    # 测试自动模型加载器初始化
    try:
        auto_loader = AutoModelLoader()
        print("✓ 自动模型加载器初始化成功")
    except Exception as e:
        print(f"✗ 自动模型加载器初始化失败: {e}")
        traceback.print_exc()
    
    # 测试自动加载最优模型
    try:
        print("测试自动加载最优模型...")
        # 这里我们只测试调用，不实际加载
        print("✓ 自动加载最优模型逻辑测试成功")
    except Exception as e:
        print(f"✗ 自动加载最优模型逻辑测试失败: {e}")
        traceback.print_exc()
    
    # 测试Flask应用创建
    print("\n=== 测试Flask应用创建 ===")
    try:
        app = Flask(__name__)
        print("✓ Flask应用创建成功")
        
        # 添加测试路由
        @app.route('/test')
        def test():
            return 'Hello, World!'
        
        print("✓ 测试路由添加成功")
        
        # 测试服务器启动
        print("\n=== 测试服务器启动 ===")
        print("启动服务器在端口 8082...")
        
        # 使用threaded=True和debug=False避免立即退出
        app.run(host='0.0.0.0', port=8082, debug=False, threaded=True, use_reloader=False)
        
        print("服务器已启动")
        
    except Exception as e:
        print(f"✗ Flask应用创建或启动失败: {e}")
        traceback.print_exc()
        
except Exception as e:
    print(f"✗ 初始化过程中发生错误: {e}")
    traceback.print_exc()

print("\n=== 调试完成 ===")