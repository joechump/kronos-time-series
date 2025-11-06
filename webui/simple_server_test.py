#!/usr/bin/env python3
"""
简单服务器测试脚本，跳过复杂模型加载，只测试Flask服务器启动
"""

import sys
import os
import traceback

# 添加项目根目录到系统路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print("=== 简单服务器测试 ===")
print(f"项目根目录: {project_root}")

# 只测试Flask应用创建和启动
try:
    from flask import Flask
    from flask_cors import CORS
    
    print("✓ Flask库导入成功")
    
    # 创建简单的Flask应用
    app = Flask(__name__)
    CORS(app)
    
    # 添加一个简单的测试路由
    @app.route('/')
    def hello():
        return 'Hello, World!'
    
    @app.route('/api/test')
    def api_test():
        from flask import jsonify
        return jsonify({'status': 'success', 'message': 'API is working'})
    
    print("✓ Flask应用创建成功")
    print("启动服务器在端口 8084...")
    
    # 启动服务器
    app.run(host='0.0.0.0', port=8084, debug=False, threaded=True, use_reloader=False)
    
    print("服务器已启动并正在运行")
    
except Exception as e:
    print(f"✗ 服务器启动失败: {e}")
    traceback.print_exc()

print("=== 测试完成 ===")