import logging
import sys
import os
import json
from logging_fix import setup_logging, fix_windows_console_encoding, ensure_utf8_encoding

def fix_log_encoding_completely():
    """彻底修复日志编码问题"""
    # 确保UTF-8编码环境
    ensure_utf8_encoding()
    
    # 修复Windows控制台编码
    fix_windows_console_encoding()
    
    # 重新配置日志系统
    setup_logging()
    
    print("日志编码已彻底修复")

def test_chinese_logging():
    """测试中文日志记录"""
    logger = logging.getLogger(__name__)
    logger.info("测试中文日志记录功能")
    logger.info("轻量级模型，适合快速预测")
    logger.info("小型模型，平衡性能与精度")
    logger.info("基础模型，提供较高精度")
    print("中文日志测试完成")

if __name__ == "__main__":
    fix_log_encoding_completely()
    test_chinese_logging()