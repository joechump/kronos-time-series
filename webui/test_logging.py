import logging
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_logging.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 测试日志记录
logger.info("这是一个测试日志信息")
logger.warning("这是一个警告日志信息")
logger.error("这是一个错误日志信息")

print("测试日志已写入")