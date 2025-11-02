import sys
import logging
import os
import io
import locale

class UTF8FileHandler(logging.FileHandler):
    """自定义文件处理器，确保使用UTF-8编码写入日志"""
    def __init__(self, filename, mode='a', encoding='utf-8', delay=False):
        # 显式指定UTF-8编码
        super().__init__(filename, mode, encoding, delay)
    
    def emit(self, record):
        """发出日志记录"""
        try:
            # 格式化日志消息
            msg = self.format(record)
            # 确保消息是字符串类型
            if not isinstance(msg, str):
                msg = str(msg)
            
            # 手动处理编码以确保正确性
            try:
                # 尝试使用UTF-8编码消息
                msg_bytes = msg.encode('utf-8')
                msg_utf8 = msg_bytes.decode('utf-8')
            except UnicodeError:
                # 如果编码失败，使用错误处理
                msg_utf8 = msg.encode('utf-8', errors='replace').decode('utf-8')
            
            # 写入日志文件
            stream = self.stream
            stream.write(msg_utf8 + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

def setup_logging():
    """设置日志系统，确保在所有环境下正确处理中文编码"""
    # 设置环境变量以支持UTF-8
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUTF8'] = '1'
    
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
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()
    
    # 添加新的处理器
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # 记录日志系统初始化信息
    root_logger.info("日志系统已初始化，使用UTF-8编码")
    
    return root_logger

def fix_windows_console_encoding():
    """修复Windows控制台编码问题"""
    if sys.platform.startswith('win'):
        # 尝试设置控制台代码页为UTF-8
        try:
            import ctypes
            # 设置控制台输出代码页为UTF-8 (65001)
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
        
        # 如果在Windows上，确保使用正确的编码
        if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding != 'utf-8':
            try:
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
                sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
            except Exception:
                pass

def ensure_utf8_encoding():
    """确保Python环境使用UTF-8编码"""
    # 设置默认编码为UTF-8
    try:
        # 尝试设置默认编码
        if hasattr(sys, 'setdefaultencoding'):
            sys.setdefaultencoding('utf-8')
    except Exception:
        pass
    
    # 设置环境变量
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUTF8'] = '1'
    os.environ['PYTHONLEGACYWINDOWSFSENCODING'] = '0'
    
    # 尝试设置locale
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except Exception:
        try:
            locale.setlocale(locale.LC_ALL, 'C.UTF-8')
        except Exception:
            pass  # 如果设置失败，继续使用默认设置
    
    # 打印当前编码设置用于调试
    print(f"系统默认编码: {sys.getdefaultencoding()}")
    print(f"文件系统编码: {sys.getfilesystemencoding()}")
    print(f"stdout编码: {getattr(sys.stdout, 'encoding', 'unknown')}")
    print(f"stderr编码: {getattr(sys.stderr, 'encoding', 'unknown')}")