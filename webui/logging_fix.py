import sys
import logging
import os
import codecs

class UTF8FileHandler(logging.FileHandler):
    """自定义文件处理器，确保使用UTF-8编码写入日志"""
    def __init__(self, filename, mode='a', encoding='utf-8', delay=False):
        # 确保使用UTF-8编码
        super().__init__(filename, mode, encoding, delay)
    
    def emit(self, record):
        """发出日志记录"""
        try:
            msg = self.format(record)
            stream = self.stream
            # 确保使用UTF-8编码写入
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

def setup_logging():
    """设置日志系统，确保在Windows环境下正确处理中文编码"""
    # 设置环境变量以支持UTF-8
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    # 配置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 创建自定义文件处理器，确保使用UTF-8编码
    file_handler = UTF8FileHandler('kronos_app.log', encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # 获取根日志记录器并配置
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 清除现有的处理器
    root_logger.handlers.clear()
    
    # 添加新的处理器
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger

def fix_windows_console_encoding():
    """修复Windows控制台编码问题"""
    if sys.platform.startswith('win'):
        # 尝试设置控制台代码页为UTF-8
        try:
            import ctypes
            # 尝试设置控制台输入和输出代码页为UTF-8 (65001)
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except Exception:
            pass  # 如果设置失败，继续使用默认设置
        
        # 确保stdout和stderr使用UTF-8编码
        if hasattr(sys.stdout, 'reconfigure'):
            try:
                sys.stdout.reconfigure(encoding='utf-8')
                sys.stderr.reconfigure(encoding='utf-8')
            except Exception:
                pass  # 如果重新配置失败，继续使用默认设置

def ensure_utf8_encoding():
    """确保Python环境使用UTF-8编码"""
    # 设置默认编码为UTF-8
    if hasattr(sys, 'setdefaultencoding'):
        try:
            sys.setdefaultencoding('utf-8')
        except Exception:
            pass
    
    # 设置文件系统编码
    os.environ['PYTHONIOENCODING'] = 'utf-8'