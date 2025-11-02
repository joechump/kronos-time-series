import sys
import logging
import os
import codecs
from logging.handlers import RotatingFileHandler

def setup_enhanced_logging():
    """增强版日志配置，专门解决Windows环境下的中文编码问题"""
    # 设置环境变量以支持UTF-8
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    # 清除现有的日志处理器
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 配置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 创建文件处理器，明确指定UTF-8编码
    # 使用codecs.open确保文件以UTF-8编码打开
    file_handler = logging.FileHandler('kronos_app.log', mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # 获取根日志记录器并配置
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 添加处理器
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger

def fix_windows_encoding():
    """修复Windows编码问题"""
    if sys.platform.startswith('win'):
        # 设置环境变量
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        
        # 尝试设置控制台代码页为UTF-8
        try:
            import ctypes
            # 设置控制台输入和输出代码页为UTF-8 (65001)
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except Exception:
            pass
        
        # 确保stdout和stderr使用UTF-8编码
        if hasattr(sys.stdout, 'reconfigure'):
            try:
                sys.stdout.reconfigure(encoding='utf-8')
                sys.stderr.reconfigure(encoding='utf-8')
            except Exception:
                pass

def test_enhanced_logging():
    """测试增强版日志功能"""
    # 应用修复
    fix_windows_encoding()
    
    # 设置增强版日志
    logger = setup_enhanced_logging()
    
    # 测试中文日志记录
    logger.info("增强版日志系统测试")
    logger.info("轻量级模型，适合快速预测")
    logger.info("小型模型，平衡性能与精度")
    logger.info("基础模型，提供较高精度")
    
    print("增强版日志测试完成")

if __name__ == "__main__":
    test_enhanced_logging()