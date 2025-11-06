#!/usr/bin/env python3
"""
简单的waitress测试
"""

from flask import Flask
from waitress import serve
import sys
import time
import threading

# 创建一个简单的Flask应用
app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, World! Server is running with Waitress.'

def server_thread():
    """在单独线程中运行服务器"""
    print("🚀 启动简单Waitress测试服务器...")
    print("🌐 访问地址: http://localhost:7070")
    print("💡 按 Ctrl+C 停止服务器")
    try:
        serve(app, host='0.0.0.0', port=7070)
    except Exception as e:
        print(f"服务器错误: {e}")

if __name__ == '__main__':
    # 在单独的线程中启动服务器
    t = threading.Thread(target=server_thread, daemon=True)
    t.start()
    
    print("⏳ 服务器线程已启动...")
    
    try:
        # 保持主线程运行
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n服务器已停止")
        sys.exit(0)