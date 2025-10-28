"""
AutoModelLoader - 智能模型自动加载系统

功能：
1. 自动检测系统资源（CPU、GPU、内存）
2. 根据系统配置智能选择最优模型
3. 提供高级用户手动配置选项
4. 实现模型加载状态管理
"""

import psutil
import torch
import os
import json
from typing import Dict, Tuple, Optional


class AutoModelLoader:
    """自动模型加载器"""
    
    def __init__(self):
        self.system_info = self._get_system_info()
        self.model_configs = self._get_model_configs()
        
    def _get_system_info(self) -> Dict:
        """获取系统资源信息"""
        try:
            # CPU信息
            cpu_count = psutil.cpu_count()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 内存信息
            memory = psutil.virtual_memory()
            memory_total_gb = round(memory.total / (1024**3), 2)
            memory_available_gb = round(memory.available / (1024**3), 2)
            
            # GPU信息
            gpu_info = self._get_gpu_info()
            
            return {
                'cpu': {
                    'count': cpu_count,
                    'usage_percent': cpu_percent,
                    'cores': cpu_count
                },
                'memory': {
                    'total_gb': memory_total_gb,
                    'available_gb': memory_available_gb,
                    'usage_percent': memory.percent
                },
                'gpu': gpu_info
            }
        except Exception as e:
            print(f"警告: 获取系统信息失败: {e}")
            return {
                'cpu': {'count': 4, 'usage_percent': 50, 'cores': 4},
                'memory': {'total_gb': 8, 'available_gb': 4, 'usage_percent': 50},
                'gpu': {'available': False}
            }
    
    def _get_gpu_info(self) -> Dict:
        """获取GPU信息"""
        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_info = []
                
                for i in range(gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    memory_total = round(props.total_memory / (1024**3), 2)
                    memory_allocated = round(torch.cuda.memory_allocated(i) / (1024**3), 2)
                    memory_cached = round(torch.cuda.memory_reserved(i) / (1024**3), 2)
                    
                    gpu_info.append({
                        'name': props.name,
                        'memory_total_gb': memory_total,
                        'memory_allocated_gb': memory_allocated,
                        'memory_cached_gb': memory_cached,
                        'compute_capability': f"{props.major}.{props.minor}"
                    })
                
                return {
                    'available': True,
                    'count': gpu_count,
                    'devices': gpu_info
                }
            else:
                return {'available': False}
        except Exception as e:
            print(f"警告: 获取GPU信息失败: {e}")
            return {'available': False}
    
    def _get_model_configs(self) -> Dict:
        """获取模型配置信息"""
        return {
            'kronos-mini': {
                'name': 'Kronos-mini',
                'model_id': 'NeoQuasar/Kronos-mini',
                'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-2k',
                'context_length': 2048,
                'params': '4.1M',
                'description': '轻量级模型，适合快速预测',
                'min_memory_gb': 2,
                'recommended_cpu_cores': 2,
                'gpu_required': False,
                'simulation_mode': True  # 标记为模拟模式
            },
            'kronos-small': {
                'name': 'Kronos-small',
                'model_id': 'NeoQuasar/Kronos-small',
                'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-2k',
                'context_length': 2048,
                'params': '14.3M',
                'description': '小型模型，平衡性能与精度',
                'min_memory_gb': 4,
                'recommended_cpu_cores': 4,
                'gpu_required': False,
                'simulation_mode': True  # 标记为模拟模式
            },
            'kronos-base': {
                'name': 'Kronos-base',
                'model_id': 'NeoQuasar/Kronos-base',
                'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-2k',
                'context_length': 2048,
                'params': '43.8M',
                'description': '基础模型，提供较高精度',
                'min_memory_gb': 8,
                'recommended_cpu_cores': 8,
                'gpu_required': True,
                'simulation_mode': True  # 标记为模拟模式
            }
        }
    
    def select_optimal_model(self) -> Tuple[str, Dict]:
        """选择最优模型"""
        system_info = self.system_info
        
        # 评估系统能力
        system_score = self._calculate_system_score(system_info)
        
        # 根据系统评分选择模型
        if system_score >= 80:
            # 高性能系统，选择kronos-base
            if self._can_run_model('kronos-base'):
                return 'kronos-base', self.model_configs['kronos-base']
        elif system_score >= 60:
            # 中等性能系统，选择kronos-small
            if self._can_run_model('kronos-small'):
                return 'kronos-small', self.model_configs['kronos-small']
        
        # 默认选择kronos-mini
        return 'kronos-mini', self.model_configs['kronos-mini']
    
    def select_optimal_device(self) -> str:
        """选择最优计算设备"""
        system_info = self.system_info
        
        # 检查GPU可用性
        if system_info['gpu'].get('available', False):
            gpu_devices = system_info['gpu'].get('devices', [])
            if gpu_devices:
                # 选择内存最大的GPU
                best_gpu = max(gpu_devices, key=lambda x: x['memory_total_gb'])
                if best_gpu['memory_total_gb'] >= 4:  # 至少4GB显存
                    return 'cuda:0'
        
        # 检查MPS（Apple Silicon）
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        
        # 默认使用CPU
        return 'cpu'
    
    def _calculate_system_score(self, system_info: Dict) -> float:
        """计算系统性能评分"""
        score = 0
        
        # CPU评分（权重40%）
        cpu_score = min(system_info['cpu']['cores'] / 8 * 100, 100)
        score += cpu_score * 0.4
        
        # 内存评分（权重40%）
        memory_score = min(system_info['memory']['total_gb'] / 16 * 100, 100)
        score += memory_score * 0.4
        
        # GPU评分（权重20%）
        if system_info['gpu'].get('available', False):
            gpu_devices = system_info['gpu'].get('devices', [])
            if gpu_devices:
                best_gpu = max(gpu_devices, key=lambda x: x['memory_total_gb'])
                gpu_score = min(best_gpu['memory_total_gb'] / 8 * 100, 100)
                score += gpu_score * 0.2
        
        return round(score, 2)
    
    def _can_run_model(self, model_key: str) -> bool:
        """检查系统是否能运行指定模型"""
        model_config = self.model_configs.get(model_key)
        if not model_config:
            return False
        
        system_info = self.system_info
        
        # 检查内存要求
        if system_info['memory']['total_gb'] < model_config['min_memory_gb']:
            return False
        
        # 检查CPU要求
        if system_info['cpu']['cores'] < model_config['recommended_cpu_cores']:
            return False
        
        # 检查GPU要求
        if model_config['gpu_required'] and not system_info['gpu'].get('available', False):
            return False
        
        return True
    
    def get_system_report(self) -> Dict:
        """获取系统资源报告"""
        return {
            'system_info': self.system_info,
            'system_score': self._calculate_system_score(self.system_info),
            'available_models': self.model_configs,
            'recommended_model': self.select_optimal_model()[0],
            'recommended_device': self.select_optimal_device()
        }
    
    def validate_model_selection(self, model_key: str, device: str) -> Dict:
        """验证模型和设备选择是否合理"""
        model_config = self.model_configs.get(model_key)
        if not model_config:
            return {'valid': False, 'reason': f'模型 {model_key} 不存在'}
        
        # 检查模型运行能力
        if not self._can_run_model(model_key):
            return {
                'valid': False,
                'reason': f'系统资源不足，无法运行 {model_config["name"]}'
            }
        
        # 检查设备兼容性
        if device.startswith('cuda') and not self.system_info['gpu'].get('available', False):
            return {'valid': False, 'reason': '选择了GPU设备但系统无可用GPU'}
        
        if device == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            return {'valid': False, 'reason': '选择了MPS设备但系统不支持'}
        
        return {'valid': True, 'reason': '选择合理'}


def create_auto_model_loader() -> AutoModelLoader:
    """创建自动模型加载器实例"""
    return AutoModelLoader()


if __name__ == "__main__":
    # 测试自动模型加载器
    loader = AutoModelLoader()
    report = loader.get_system_report()
    
    print("=== 系统资源报告 ===")
    print(f"系统评分: {report['system_score']}")
    print(f"推荐模型: {report['recommended_model']}")
    print(f"推荐设备: {report['recommended_device']}")
    
    print("\n=== 系统详情 ===")
    print(f"CPU核心数: {report['system_info']['cpu']['cores']}")
    print(f"内存总量: {report['system_info']['memory']['total_gb']} GB")
    print(f"GPU可用: {report['system_info']['gpu'].get('available', False)}")
    
    print("\n=== 模型验证 ===")
    for model_key in ['kronos-mini', 'kronos-small', 'kronos-base']:
        validation = loader.validate_model_selection(model_key, 'cpu')
        print(f"{model_key} on CPU: {validation['reason']}")