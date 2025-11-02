import sys
import logging
import os
from logging_fix import setup_logging, fix_windows_console_encoding, ensure_utf8_encoding

# 确保UTF-8编码
ensure_utf8_encoding()

# 修复Windows控制台编码问题
fix_windows_console_encoding()

# 配置日志系统
logger = setup_logging()

# 测试中文日志记录
logger.info("这是一个测试中文日志消息")
logger.info("模型加载成功")
logger.info("轻量级模型，适合快速预测")
logger.info("小型模型，性能和精度平衡")
logger.info("基础模型，提供较高精度")

print("系统默认编码:", sys.getdefaultencoding())
print("stdout编码:", getattr(sys.stdout, 'encoding', 'Unknown'))
print("stderr编码:", getattr(sys.stderr, 'encoding', 'Unknown'))

# 检查环境变量
print("PYTHONIOENCODING:", os.environ.get('PYTHONIOENCODING', 'Not set'))