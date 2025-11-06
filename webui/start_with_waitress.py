#!/usr/bin/env python3
"""
使用waitress启动Kronos Web UI服务器
"""

import os
import sys
import time
import threading

def keep_alive():
    """保持主线程活跃"""
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n收到停止信号，正在关闭服务器...")
        sys.exit(0)

def main():
    """主函数"""
    print("🚀 正在启动 Kronos Web UI (使用 Waitress)...")
    print("=" * 50)
    
    # 添加项目路径到sys.path
    project_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_path)
    
    try:
        # 导入Flask应用
        from app import app
        
        # 使用waitress启动服务器
        from waitress import serve
        
        # 在单独的线程中启动服务器
        def start_server():
            try:
                print("✅ Web服务器启动成功!")
                print(f"🌐 访问地址: http://localhost:7070")
                print("💡 提示: 按 Ctrl+C 停止服务器")
                
                # 启动服务器
                print("正在启动服务器...")
                serve(app, host='0.0.0.0', port=7070, threads=4)
            except Exception as e:
                print(f"❌ 服务器运行错误: {e}")
                import traceback
                traceback.print_exc()
        
        server_thread = threading.Thread(target=start_server, daemon=True)
        server_thread.start()
        
        print("⏳ 服务器线程已启动...")
        
        # 保持主线程活跃
        keep_alive()
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保所有依赖已安装")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()