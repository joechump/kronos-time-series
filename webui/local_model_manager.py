"""
本地模型管理器 - Local Model Manager

功能：
1. 管理本地模型存储目录结构
2. 提供模型下载、缓存、加载功能
3. 支持模型版本管理
4. 实现网络失败时的降级方案
"""

import os
import json
import shutil
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalModelManager:
    """本地模型管理器"""
    
    def __init__(self, base_dir: str = None):
        """初始化本地模型管理器"""
        if base_dir is None:
            base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'local_models')
        
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / 'models'
        self.cache_dir = self.base_dir / 'cache'
        self.metadata_file = self.base_dir / 'models_metadata.json'
        
        # 创建必要的目录结构
        self._create_directories()
        
        # 加载模型元数据
        self.models_metadata = self._load_metadata()
    
    def _create_directories(self):
        """创建必要的目录结构"""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_metadata(self) -> Dict:
        """加载模型元数据"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
        
        return {
            'models': {},
            'last_updated': None,
            'version': '1.0'
        }
    
    def _save_metadata(self):
        """保存模型元数据"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.models_metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def get_model_path(self, model_id: str) -> Path:
        """获取模型本地路径"""
        # 将模型ID转换为安全的文件名
        safe_name = model_id.replace('/', '_')
        return self.models_dir / safe_name
    
    def is_model_available_locally(self, model_id: str) -> bool:
        """检查模型是否在本地可用"""
        model_path = self.get_model_path(model_id)
        
        # 检查模型目录是否存在且包含必要的文件
        if not model_path.exists():
            return False
        
        # 检查是否包含模型文件（这里需要根据实际模型文件结构调整）
        required_files = ['config.json', 'pytorch_model.bin']
        for file in required_files:
            if not (model_path / file).exists():
                return False
        
        return True
    
    def download_model(self, model_id: str, tokenizer_id: str = None) -> Tuple[bool, str]:
        """下载模型到本地"""
        if tokenizer_id is None:
            tokenizer_id = model_id
        
        try:
            from transformers import AutoModel, AutoTokenizer
            import os
            
            model_path = self.get_model_path(model_id)
            
            # 如果模型已存在，先删除旧版本
            if model_path.exists():
                shutil.rmtree(model_path)
            
            # 创建模型目录
            model_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Downloading model: {model_id}")
            
            # 设置镜像源和重试机制
            original_hf_endpoint = os.environ.get('HF_ENDPOINT', '')
            
            # 尝试不同的下载策略
            download_strategies = [
                {'name': '直接下载', 'hf_endpoint': ''},
                {'name': '镜像源1', 'hf_endpoint': 'https://hf-mirror.com'},
                {'name': '镜像源2', 'hf_endpoint': 'https://hf-mirror.com'}
            ]
            
            last_error = None
            
            for strategy in download_strategies:
                try:
                    # 设置镜像源
                    if strategy['hf_endpoint']:
                        os.environ['HF_ENDPOINT'] = strategy['hf_endpoint']
                        logger.info(f"尝试使用{strategy['name']}: {strategy['hf_endpoint']}")
                    
                    # 下载模型（增加超时时间）
                    model = AutoModel.from_pretrained(
                        model_id, 
                        cache_dir=str(self.cache_dir),
                        local_files_only=False,
                        force_download=True
                    )
                    model.save_pretrained(str(model_path))
                    
                    # 下载tokenizer
                    tokenizer = AutoTokenizer.from_pretrained(
                        tokenizer_id, 
                        cache_dir=str(self.cache_dir),
                        local_files_only=False,
                        force_download=True
                    )
                    tokenizer.save_pretrained(str(model_path))
                    
                    # 恢复原始环境变量
                    if original_hf_endpoint:
                        os.environ['HF_ENDPOINT'] = original_hf_endpoint
                    else:
                        os.environ.pop('HF_ENDPOINT', None)
                    
                    # 更新元数据
                    self.models_metadata['models'][model_id] = {
                        'local_path': str(model_path),
                        'downloaded_at': str(Path(model_path).stat().st_ctime),
                        'size': self._get_directory_size(model_path),
                        'tokenizer_id': tokenizer_id
                    }
                    self.models_metadata['last_updated'] = str(Path(model_path).stat().st_ctime)
                    self._save_metadata()
                    
                    logger.info(f"Model downloaded successfully: {model_id}")
                    return True, "下载成功"
                    
                except Exception as e:
                    last_error = e
                    logger.warning(f"{strategy['name']}失败: {e}")
                    
                    # 清理部分下载的文件
                    if model_path.exists():
                        shutil.rmtree(model_path)
                    model_path.mkdir(parents=True, exist_ok=True)
                    
                    # 短暂延迟后重试
                    import time
                    time.sleep(2)
            
            # 恢复原始环境变量
            if original_hf_endpoint:
                os.environ['HF_ENDPOINT'] = original_hf_endpoint
            else:
                os.environ.pop('HF_ENDPOINT', None)
            
            # 所有策略都失败
            logger.error(f"所有下载策略都失败: {last_error}")
            return False, f"下载失败: {str(last_error)}"
            
        except Exception as e:
            logger.error(f"Failed to download model {model_id}: {e}")
            return False, f"下载失败: {str(e)}"
    
    def _get_directory_size(self, path: Path) -> int:
        """计算目录大小"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        return total_size
    
    def load_model_from_local(self, model_id: str):
        """从本地加载模型"""
        if not self.is_model_available_locally(model_id):
            raise FileNotFoundError(f"Model {model_id} not available locally")
        
        try:
            from transformers import AutoModel, AutoTokenizer
            
            model_path = self.get_model_path(model_id)
            
            logger.info(f"Loading model from local: {model_id}")
            
            # 从本地加载模型
            model = AutoModel.from_pretrained(str(model_path))
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            
            logger.info(f"Model loaded successfully from local: {model_id}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model from local {model_id}: {e}")
            raise
    
    def get_available_local_models(self) -> List[Dict]:
        """获取本地可用的模型列表"""
        available_models = []
        
        for model_id, metadata in self.models_metadata.get('models', {}).items():
            if self.is_model_available_locally(model_id):
                available_models.append({
                    'model_id': model_id,
                    'local_path': metadata.get('local_path'),
                    'downloaded_at': metadata.get('downloaded_at'),
                    'size': metadata.get('size', 0),
                    'tokenizer_id': metadata.get('tokenizer_id', model_id)
                })
        
        return available_models
    
    def cleanup_cache(self):
        """清理缓存目录"""
        try:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True)
                logger.info("Cache cleaned successfully")
        except Exception as e:
            logger.error(f"Failed to clean cache: {e}")
    
    def get_storage_info(self) -> Dict:
        """获取存储信息"""
        total_size = 0
        model_count = 0
        
        for model_id in self.models_metadata.get('models', {}):
            if self.is_model_available_locally(model_id):
                model_count += 1
                model_path = self.get_model_path(model_id)
                total_size += self._get_directory_size(model_path)
        
        return {
            'total_models': model_count,
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'models_dir': str(self.models_dir),
            'cache_dir': str(self.cache_dir)
        }


def create_local_model_manager() -> LocalModelManager:
    """创建本地模型管理器实例"""
    return LocalModelManager()


if __name__ == "__main__":
    # 测试本地模型管理器
    manager = LocalModelManager()
    
    print("=== 本地模型管理器测试 ===")
    print(f"基础目录: {manager.base_dir}")
    print(f"模型目录: {manager.models_dir}")
    print(f"缓存目录: {manager.cache_dir}")
    
    # 测试存储信息
    storage_info = manager.get_storage_info()
    print(f"存储信息: {storage_info}")
    
    # 测试可用模型
    available_models = manager.get_available_local_models()
    print(f"本地可用模型: {len(available_models)} 个")
    
    for model_info in available_models:
        print(f"  - {model_info['model_id']}")
    
    print("=== 测试完成 ===")