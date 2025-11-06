#!/usr/bin/env python3
"""
测试模型加载状态
"""

import sys
import os

# 添加路径
sys.path.append('.')

def test_model_loading():
    """测试模型加载状态"""
    print("=== 测试模型加载状态 ===")
    
    try:
        from direct_model_loader import DirectModelLoader
        print("✅ DirectModelLoader导入成功")
        
        # 创建模型加载器
        loader = DirectModelLoader()
        print("✅ DirectModelLoader初始化成功")
        
        # 检查可用模型
        available_models = loader.available_models
        print(f"可用模型: {list(available_models.keys())}")
        
        # 检查每个模型的详细信息
        for model_key, model_info in available_models.items():
            print(f"\n模型 {model_key}:")
            print(f"  状态: {model_info.get('status', 'unknown')}")
            print(f"  路径: {model_info.get('local_path', 'unknown')}")
            
            # 检查必要的文件是否存在
            if model_info.get('local_path'):
                import os
                config_file = os.path.join(model_info['local_path'], 'config.json')
                model_file = os.path.join(model_info['local_path'], 'model.safetensors')
                
                print(f"  配置文件存在: {os.path.exists(config_file)}")
                print(f"  模型文件存在: {os.path.exists(model_file)}")
        
        # 检查已加载的模型
        loaded_model = loader.get_loaded_model()
        print(f"\n已加载模型: {loaded_model is not None}")
        
        if loaded_model:
            print("✅ 模型已加载成功")
            print(f"模型名称: {loaded_model.get('model_name', 'unknown')}")
            print(f"模型路径: {loaded_model.get('model_path', 'unknown')}")
            print(f"设备信息: {loaded_model.get('device', 'unknown')}")
        else:
            print("❌ 模型未加载")
            
            # 尝试自动加载最优模型
            print("\n=== 尝试自动加载最优模型 ===")
            success, message, model_key = loader.auto_load_best_model()
            print(f"自动加载结果: {success}")
            print(f"消息: {message}")
            print(f"模型键: {model_key}")
            
            if success:
                loaded_model = loader.get_loaded_model()
                print(f"重新检查已加载模型: {loaded_model is not None}")
                if loaded_model:
                    print("✅ 自动加载成功")
                else:
                    print("❌ 自动加载失败")
        
        return loaded_model is not None
        
    except Exception as e:
        import traceback
        print(f"❌ 测试失败: {str(e)}")
        print(f"详细错误信息:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("开始测试模型加载状态...")
    
    success = test_model_loading()
    
    if success:
        print("\n✅ 模型加载状态测试通过！")
    else:
        print("\n❌ 模型加载状态测试失败！")
    
    print("\n=== 检查app.py中的模型加载状态 ===")
    
    # 检查app.py中的模型加载逻辑
    try:
        # 模拟app.py中的模型加载逻辑
        from direct_model_loader import DirectModelLoader
        
        direct_model_loader = DirectModelLoader()
        print("直接模型加载器初始化成功")
        
        # 自动加载最优模型
        success, message, model_key = direct_model_loader.auto_load_best_model()
        if success:
            print(f"自动加载模型: {model_key} - {message}")
            # 设置全局模型变量
            loaded_model = direct_model_loader.get_loaded_model()
            if loaded_model:
                print(f"模型 {model_key} 加载成功")
                print(f"模型检查: direct_model_loader.get_loaded_model() = {loaded_model is not None}")
            else:
                print("❌ 模型加载失败")
        else:
            print(f"自动加载失败: {message}")
            
    except Exception as e:
        import traceback
        print(f"❌ app.py模型加载逻辑测试失败: {str(e)}")
        print(f"详细错误信息:\n{traceback.format_exc()}")