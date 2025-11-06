#!/usr/bin/env python3
"""
调试服务器启动问题的脚本，检查是否有异常导致服务器退出
"""

import sys
import os
import traceback

# 添加项目根目录到系统路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.append(os.path.join(project_root, 'model'))

print("=== 调试服务器启动问题 ===")
print(f"项目根目录: {project_root}")

# 尝试导入并执行app.py中的初始化逻辑
print("\n=== 执行app.py初始化逻辑 ===")

try:
    # 模拟app.py中的导入和初始化
    import pandas as pd
    import numpy as np
    import json
    import plotly.graph_objects as go
    import plotly.utils
    from flask import Flask, render_template, request, jsonify
    from flask_cors import CORS
    import warnings
    import datetime
    import argparse
    import logging
    warnings.filterwarnings('ignore')
    
    print("✓ 基础库导入成功")
    
    # 导入并应用日志修复模块
    try:
        from logging_fix_robust import setup_logging, fix_windows_console_encoding, ensure_utf8_encoding
        ensure_utf8_encoding()
        fix_windows_console_encoding()
        setup_logging()
        logger = logging.getLogger(__name__)
        print("✓ 日志系统配置成功")
    except Exception as e:
        print(f"✗ 日志系统配置失败: {e}")
        traceback.print_exc()
    
    # 测试数据提供器初始化
    try:
        from akshare_data_provider import AkshareDataProvider
        data_provider = AkshareDataProvider()
        print("✓ 数据提供器初始化成功")
    except Exception as e:
        print(f"✗ 数据提供器初始化失败: {e}")
        traceback.print_exc()
    
    # 测试自动模型加载器初始化
    try:
        from auto_model_loader import AutoModelLoader
        auto_loader = AutoModelLoader()
        print("✓ 自动模型加载器初始化成功")
    except Exception as e:
        print(f"✗ 自动模型加载器初始化失败: {e}")
        traceback.print_exc()
    
    # 测试本地模型管理器初始化
    try:
        from local_model_manager import LocalModelManager
        local_model_manager = LocalModelManager()
        print("✓ 本地模型管理器初始化成功")
    except Exception as e:
        print(f"✗ 本地模型管理器初始化失败: {e}")
        traceback.print_exc()
    
    # 测试直接模型加载器初始化
    try:
        from direct_model_loader import DirectModelLoader
        direct_model_loader = DirectModelLoader()
        print("✓ 直接模型加载器初始化成功")
        
        # 测试自动加载最优模型
        try:
            success, message, model_key = direct_model_loader.auto_load_best_model()
            if success:
                print(f"✓ 自动加载模型成功: {model_key} - {message}")
                
                # 测试获取加载的模型
                loaded_model = direct_model_loader.get_loaded_model()
                if loaded_model:
                    print("✓ 获取加载的模型成功")
                    
                    # 测试创建预测器
                    try:
                        from model import KronosPredictor
                        predictor = KronosPredictor(loaded_model['model'], loaded_model['tokenizer'])
                        print("✓ 预测器创建成功")
                    except Exception as e:
                        print(f"✗ 预测器创建失败: {e}")
                        traceback.print_exc()
                else:
                    print("✗ 获取加载的模型失败")
            else:
                print(f"✗ 自动加载模型失败: {message}")
        except Exception as e:
            print(f"✗ 自动加载模型过程中发生错误: {e}")
            traceback.print_exc()
            
    except Exception as e:
        print(f"✗ 直接模型加载器初始化失败: {e}")
        traceback.print_exc()
    
    # 测试模型库可用性
    try:
        from model import Kronos, KronosTokenizer, KronosPredictor
        # 测试模型库是否真正可用
        try:
            test_tokenizer = KronosTokenizer.from_pretrained('NeoQuasar/Kronos-Tokenizer-base')
            MODEL_AVAILABLE = True
            print("✓ 模型库可用性测试成功")
        except Exception as e:
            MODEL_AVAILABLE = False
            print(f"✗ 模型库可用性测试失败: {e}")
    except ImportError:
        MODEL_AVAILABLE = False
        print("✗ 无法导入Kronos模型")
    
    # 测试Flask应用创建
    print("\n=== 测试Flask应用创建 ===")
    try:
        app = Flask(__name__)
        CORS(app, resources={r"/api/*": {"origins": "*"}})
        print("✓ Flask应用创建成功")
        
        # 添加测试路由
        @app.route('/test')
        def test():
            return 'Hello, World!'
        
        print("✓ 测试路由添加成功")
        
        # 测试服务器启动
        print("\n=== 测试服务器启动 ===")
        print("启动服务器在端口 8083...")
        
        # 使用threaded=True和debug=False避免立即退出
        app.run(host='0.0.0.0', port=8083, debug=False, threaded=True, use_reloader=False)
        
        print("服务器已启动")
        
    except Exception as e:
        print(f"✗ Flask应用创建或启动失败: {e}")
        traceback.print_exc()
        
except Exception as e:
    print(f"✗ 初始化过程中发生错误: {e}")
    traceback.print_exc()

print("\n=== 调试完成 ===")