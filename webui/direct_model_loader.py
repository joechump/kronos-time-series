"""
DirectModelLoader - 直接模型加载器

功能：
1. 直接从指定路径加载已下载的模型
2. 自动检测可用的模型
3. 提供模型加载状态管理
"""

import os
import json
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple


class DirectModelLoader:
    """直接模型加载器"""
    
    def __init__(self, models_dir: str = None):
        """初始化模型加载器"""
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        
        self.models_dir = Path(models_dir)
        self.loaded_models = {}
        self.available_models = self._scan_available_models()
        
    def _scan_available_models(self) -> Dict:
        """扫描可用的模型"""
        available_models = {}
        
        if not self.models_dir.exists():
            return available_models
        
        # 模型配置映射
        model_configs = {
            'models--NeoQuasar--Kronos-mini': {
                'key': 'kronos-mini',
                'name': 'Kronos-mini',
                'model_id': 'NeoQuasar/Kronos-mini',
                'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-2k',
                'context_length': 2048,
                'params': '4.1M',
                'description': '轻量级模型，适合快速预测'
            },
            'models--NeoQuasar--Kronos-small': {
                'key': 'kronos-small',
                'name': 'Kronos-small',
                'model_id': 'NeoQuasar/Kronos-small',
                'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
                'context_length': 512,
                'params': '24.7M',
                'description': '小型模型，平衡性能与精度'
            },
            'models--NeoQuasar--Kronos-base': {
                'key': 'kronos-base',
                'name': 'Kronos-base',
                'model_id': 'NeoQuasar/Kronos-base',
                'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
                'context_length': 512,
                'params': '102.3M',
                'description': '基础模型，提供较高精度'
            }
        }
        
        for model_dir_name, config in model_configs.items():
            model_dir = self.models_dir / model_dir_name
            if model_dir.exists():
                # 检查snapshots目录
                snapshots_dir = model_dir / 'snapshots'
                if snapshots_dir.exists():
                    # 获取第一个snapshot
                    snapshots = list(snapshots_dir.iterdir())
                    if snapshots:
                        snapshot_path = snapshots[0]
                        # 检查必要的文件
                        required_files = ['config.json', 'model.safetensors']
                        if all((snapshot_path / file).exists() for file in required_files):
                            available_models[config['key']] = {
                                **config,
                                'local_path': str(snapshot_path),
                                'status': 'available'
                            }
                        else:
                            # 记录缺失的文件
                            missing_files = [file for file in required_files if not (snapshot_path / file).exists()]
                            print(f"模型 {config['key']} 缺少文件: {missing_files}")
        
        print(f"扫描完成，可用模型: {list(available_models.keys())}")
        return available_models
    
    def load_model(self, model_key: str) -> Tuple[bool, str]:
        """加载指定模型"""
        if model_key not in self.available_models:
            return False, f"模型 {model_key} 不可用"
        
        try:
            model_config = self.available_models[model_key]
            model_path = model_config['local_path']
            
            print(f"正在加载模型 {model_key}，路径: {model_path}")
            
            # 动态导入必要的模块
            import sys
            sys.path.append('..')
            from model import Kronos, KronosTokenizer
            
            # 根据模型类型确定tokenizer路径
            tokenizer_path = None
            if model_key == 'kronos-mini':
                tokenizer_path = self.models_dir / 'models--NeoQuasar--Kronos-Tokenizer-2k' / 'snapshots' / '26966d0035065a0cae0ebad7af8ece35bc1fb51c'
            elif model_key == 'kronos-small':
                # kronos-small 应该使用 Kronos-Tokenizer-2k，因为 Kronos-Tokenizer-base 不存在
                tokenizer_path = self.models_dir / 'models--NeoQuasar--Kronos-Tokenizer-2k' / 'snapshots' / '26966d0035065a0cae0ebad7af8ece35bc1fb51c'
            elif model_key == 'kronos-base':
                # kronos-base 应该使用 Kronos-Tokenizer-2k，因为 Kronos-Tokenizer-base 不存在
                tokenizer_path = self.models_dir / 'models--NeoQuasar--Kronos-Tokenizer-2k' / 'snapshots' / '26966d0035065a0cae0ebad7af8ece35bc1fb51c'
            
            # 检查tokenizer路径是否存在
            if tokenizer_path and tokenizer_path.exists():
                print(f"加载tokenizer，路径: {tokenizer_path}")
                # 从配置文件加载参数，然后手动创建KronosTokenizer实例
                import json
                with open(tokenizer_path / 'config.json', 'r', encoding='utf-8') as f:
                    tokenizer_config = json.load(f)
                
                # 使用配置参数创建KronosTokenizer实例
                tokenizer = KronosTokenizer(
                    d_in=tokenizer_config['d_in'],
                    d_model=tokenizer_config['d_model'],
                    n_heads=tokenizer_config['n_heads'],
                    ff_dim=tokenizer_config['ff_dim'],
                    n_enc_layers=tokenizer_config['n_enc_layers'],
                    n_dec_layers=tokenizer_config['n_dec_layers'],
                    ffn_dropout_p=tokenizer_config['ffn_dropout_p'],
                    attn_dropout_p=tokenizer_config['attn_dropout_p'],
                    resid_dropout_p=tokenizer_config['resid_dropout_p'],
                    s1_bits=tokenizer_config['s1_bits'],
                    s2_bits=tokenizer_config['s2_bits'],
                    beta=tokenizer_config['beta'],
                    gamma0=tokenizer_config['gamma0'],
                    gamma=tokenizer_config['gamma'],
                    zeta=tokenizer_config['zeta'],
                    group_size=tokenizer_config['group_size']
                )
                print(f"Tokenizer加载成功")
            else:
                # 如果tokenizer路径不存在，尝试从模型路径加载
                print(f"使用模型路径加载tokenizer: {model_path}")
                # 从模型路径加载配置
                import json
                with open(model_path / 'config.json', 'r', encoding='utf-8') as f:
                    model_config = json.load(f)
                
                # 使用模型配置创建KronosTokenizer实例
                tokenizer = KronosTokenizer(
                    d_in=6,  # 默认输入维度
                    d_model=model_config['d_model'],
                    n_heads=model_config['n_heads'],
                    ff_dim=model_config['ff_dim'],
                    n_enc_layers=4,  # 默认编码器层数
                    n_dec_layers=4,  # 默认解码器层数
                    ffn_dropout_p=model_config.get('ffn_dropout_p', 0.0),
                    attn_dropout_p=model_config.get('attn_dropout_p', 0.0),
                    resid_dropout_p=model_config.get('resid_dropout_p', 0.0),
                    s1_bits=model_config['s1_bits'],
                    s2_bits=model_config['s2_bits'],
                    beta=0.05,  # 默认参数
                    gamma0=1.0,  # 默认参数
                    gamma=1.1,   # 默认参数
                    zeta=0.05,   # 默认参数
                    group_size=5  # 默认参数
                )
                print(f"Tokenizer加载成功")
            
            # 加载模型
            print(f"加载模型...")
            model = Kronos.from_pretrained(model_path)
            print(f"模型加载成功")
            
            # 设置模型为评估模式
            model.eval()
            
            # 存储加载的模型
            self.loaded_models[model_key] = {
                'model': model,
                'tokenizer': tokenizer,
                'config': model_config
            }
            
            print(f"模型 {model_key} 加载成功")
            return True, f"模型 {model_key} 加载成功"
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"加载模型 {model_key} 失败: {str(e)}")
            print(f"详细错误信息: {error_details}")
            return False, f"加载模型失败: {str(e)}"
    
    def auto_load_best_model(self) -> Tuple[bool, str, str]:
        """自动加载最优模型"""
        if not self.available_models:
            return False, "没有可用的模型", ""
        
        # 按优先级尝试加载模型
        priority_order = ['kronos-base', 'kronos-small', 'kronos-mini']
        
        for model_key in priority_order:
            if model_key in self.available_models:
                success, message = self.load_model(model_key)
                if success:
                    return True, message, model_key
        
        # 如果优先级模型都失败，尝试加载任意可用模型
        for model_key in self.available_models.keys():
            success, message = self.load_model(model_key)
            if success:
                return True, message, model_key
        
        return False, "所有模型加载失败", ""
    
    def get_loaded_model(self, model_key: str = None):
        """获取已加载的模型"""
        if model_key:
            return self.loaded_models.get(model_key)
        
        # 返回第一个加载的模型
        if self.loaded_models:
            return next(iter(self.loaded_models.values()))
        
        return None
    
    def get_model_status(self) -> Dict:
        """获取模型状态"""
        return {
            'available_models': self.available_models,
            'loaded_models': list(self.loaded_models.keys()),
            'models_dir': str(self.models_dir)
        }


# 全局模型加载器实例
direct_model_loader = DirectModelLoader()