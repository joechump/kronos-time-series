import logging
import os
import sys

# 修复日志编码问题
def fix_logging_encoding():
    """修复Windows环境下日志文件的编码问题"""
    # 移除现有的日志处理器
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 重新配置日志，确保使用UTF-8编码
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('kronos_app.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

if __name__ == "__main__":
    fix_logging_encoding()
    logger = logging.getLogger(__name__)
    logger.info("日志编码已修复")
    print("日志编码修复完成")