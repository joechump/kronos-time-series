#!/usr/bin/env python3
"""
详细调试导入问题
"""

import os
import sys
import traceback

def test_sys_path():
    """测试系统路径"""
    print("=== 测试系统路径 ===")
    
    # 获取项目根目录
    current_file = os.path.abspath(__file__)
    webui_dir = os.path.dirname(current_file)
    project_root = os.path.dirname(webui_dir)
    model_dir = os.path.join(project_root, 'model')
    
    print(f"当前文件: {current_file}")
    print(f"webui目录: {webui_dir}")
    print(f"项目根目录: {project_root}")
    print(f"model目录: {model_dir}")
    
    # 检查目录是否存在
    print(f"项目根目录存在: {os.path.exists(project_root)}")
    print(f"model目录存在: {os.path.exists(model_dir)}")
    
    # 检查model目录内容
    if os.path.exists(model_dir):
        print(f"model目录内容: {os.listdir(model_dir)}")
    
    # 添加路径
    sys.path.insert(0, project_root)
    sys.path.insert(0, model_dir)
    
    print(f"系统路径:")
    for i, path in enumerate(sys.path[:10]):  # 只显示前10个路径
        print(f"  {i}: {path}")
    
    return project_root, model_dir

def test_model_import():
    """测试model模块导入"""
    print("\n=== 测试model模块导入 ===")
    
    # 尝试直接导入
    try:
        import model
        print("✅ import model 成功")
        print(f"model模块路径: {model.__file__}")
        
        # 检查模块内容
        print(f"model模块内容: {dir(model)}")
        
        # 尝试导入具体类
        try:
            from model import KronosPredictor
            print("✅ from model import KronosPredictor 成功")
        except Exception as e:
            print(f"❌ from model import KronosPredictor 失败: {e}")
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ import model 失败: {e}")
        traceback.print_exc()
        
        # 尝试其他导入方式
        print("\n=== 尝试其他导入方式 ===")
        
        # 尝试从绝对路径导入
        try:
            import importlib.util
            
            # 获取model模块的绝对路径
            current_file = os.path.abspath(__file__)
            webui_dir = os.path.dirname(current_file)
            project_root = os.path.dirname(webui_dir)
            model_init_path = os.path.join(project_root, 'model', '__init__.py')
            
            if os.path.exists(model_init_path):
                print(f"model __init__.py 存在: {model_init_path}")
                
                spec = importlib.util.spec_from_file_location("model", model_init_path)
                model_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(model_module)
                
                print("✅ 通过绝对路径导入model成功")
                print(f"模块内容: {dir(model_module)}")
                
                # 尝试获取KronosPredictor
                if hasattr(model_module, 'KronosPredictor'):
                    print("✅ 找到KronosPredictor类")
                else:
                    print("❌ 未找到KronosPredictor类")
                    
            else:
                print(f"❌ model __init__.py 不存在: {model_init_path}")
                
        except Exception as e2:
            print(f"❌ 绝对路径导入失败: {e2}")
            traceback.print_exc()

def main():
    """主测试函数"""
    print("开始详细调试导入问题...")
    
    # 测试系统路径
    project_root, model_dir = test_sys_path()
    
    # 测试model导入
    test_model_import()
    
    print("\n=== 调试完成 ===")

if __name__ == "__main__":
    main()