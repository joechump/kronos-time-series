"""
akShare数据提供器模块
为Kronos 2.0提供股票数据获取功能
"""

import akshare as ak
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import cachetools
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AkshareDataProvider:
    """akshare数据提供器类"""
    
    def __init__(self, cache_size: int = 1000, cache_ttl: int = 3600):
        """
        初始化数据提供器
        
        Args:
            cache_size: 缓存大小
            cache_ttl: 缓存过期时间（秒）
        """
        self.cache = cachetools.TTLCache(maxsize=cache_size, ttl=cache_ttl)
        self.session = requests.Session()
        
        # 配置请求头
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8'
        })
    
    def get_stock_data(self, symbol: str, period: str = 'daily', 
                      start_date: Optional[str] = None, 
                      end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取股票历史数据
        
        Args:
            symbol: 股票代码（如：000001）
            period: 数据周期（daily, weekly, monthly）
            start_date: 开始日期（YYYYMMDD）
            end_date: 结束日期（YYYYMMDD）
            
        Returns:
            pandas.DataFrame: 股票数据
        """
        cache_key = f"stock_{symbol}_{period}_{start_date}_{end_date}"
        
        # 检查缓存
        if cache_key in self.cache:
            logger.info(f"从缓存获取股票数据: {symbol}")
            return self.cache[cache_key]
        
        try:
            # 设置默认日期范围（最近1年）
            if not end_date:
                end_date = datetime.now().strftime('%Y%m%d')
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
            
            logger.info(f"获取股票数据: {symbol}, 周期: {period}, 日期范围: {start_date}-{end_date}")
            
            # 获取股票数据
            if period == 'daily':
                stock_data = ak.stock_zh_a_hist(
                    symbol=symbol, 
                    period='daily', 
                    start_date=start_date, 
                    end_date=end_date,
                    adjust="hfq"
                )
            elif period == 'weekly':
                stock_data = ak.stock_zh_a_hist(
                    symbol=symbol, 
                    period='weekly', 
                    start_date=start_date, 
                    end_date=end_date,
                    adjust="hfq"
                )
            elif period == 'monthly':
                stock_data = ak.stock_zh_a_hist(
                    symbol=symbol, 
                    period='monthly', 
                    start_date=start_date, 
                    end_date=end_date,
                    adjust="hfq"
                )
            else:
                raise ValueError(f"不支持的周期类型: {period}")
            
            # 数据清洗和格式化
            if not stock_data.empty:
                # 重命名列以保持一致性
                stock_data = stock_data.rename(columns={
                    '日期': 'date',
                    '开盘': 'open',
                    '收盘': 'close',
                    '最高': 'high',
                    '最低': 'low',
                    '成交量': 'volume',
                    '成交额': 'amount',
                    '振幅': 'amplitude',
                    '涨跌幅': 'change_rate',
                    '涨跌额': 'change_amount',
                    '换手率': 'turnover_rate'
                })
                
                # 确保日期格式正确
                stock_data['date'] = pd.to_datetime(stock_data['date'])
                stock_data = stock_data.sort_values('date').reset_index(drop=True)
                
                # 添加股票代码列
                stock_data['symbol'] = symbol
                
                # 缓存数据
                self.cache[cache_key] = stock_data
                
                logger.info(f"成功获取股票数据: {symbol}, 数据量: {len(stock_data)}")
                return stock_data
            else:
                logger.warning(f"未获取到股票数据: {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"获取股票数据失败: {symbol}, 错误: {e}")
            return pd.DataFrame()
    
    def search_stock(self, keyword: str) -> List[Dict]:
        """
        搜索股票
        
        Args:
            keyword: 搜索关键词（股票代码或名称）
            
        Returns:
            List[Dict]: 股票列表
        """
        cache_key = f"search_{keyword}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # 获取A股股票列表
            stock_list = ak.stock_zh_a_spot_em()
            
            # 搜索匹配的股票
            if keyword.isdigit():
                # 按代码搜索
                results = stock_list[stock_list['代码'].str.contains(keyword)]
            else:
                # 按名称搜索
                results = stock_list[stock_list['名称'].str.contains(keyword, case=False)]
            
            # 格式化结果
            stock_results = []
            for _, row in results.iterrows():
                stock_results.append({
                    'symbol': row['代码'],
                    'name': row['名称'],
                    'latest_price': row['最新价'],
                    'change_rate': row['涨跌幅'],
                    'change_amount': row['涨跌额'],
                    'volume': row['成交量'],
                    'amount': row['成交额']
                })
            
            self.cache[cache_key] = stock_results
            return stock_results
            
        except Exception as e:
            logger.error(f"搜索股票失败: {keyword}, 错误: {e}")
            return []
    
    def get_trading_calendar(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取交易日历
        
        Args:
            start_date: 开始日期（YYYYMMDD）
            end_date: 结束日期（YYYYMMDD）
            
        Returns:
            pandas.DataFrame: 交易日历
        """
        cache_key = f"calendar_{start_date}_{end_date}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # 获取交易日历
            trade_calendar = ak.tool_trade_date_hist_sina()
            
            # 过滤日期范围
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            trade_calendar['trade_date'] = pd.to_datetime(trade_calendar['trade_date'])
            
            filtered_calendar = trade_calendar[
                (trade_calendar['trade_date'] >= start_dt) & 
                (trade_calendar['trade_date'] <= end_dt)
            ]
            
            self.cache[cache_key] = filtered_calendar
            return filtered_calendar
            
        except Exception as e:
            logger.error(f"获取交易日历失败: {start_date}-{end_date}, 错误: {e}")
            return pd.DataFrame()
    
    def is_trading_day(self, date: str) -> bool:
        """
        判断是否为交易日
        
        Args:
            date: 日期（YYYYMMDD）
            
        Returns:
            bool: 是否为交易日
        """
        cache_key = f"trading_day_{date}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # 获取指定日期的交易日信息
            date_dt = pd.to_datetime(date)
            trade_calendar = self.get_trading_calendar(date, date)
            
            is_trading = not trade_calendar.empty
            self.cache[cache_key] = is_trading
            return is_trading
            
        except Exception as e:
            logger.error(f"判断交易日失败: {date}, 错误: {e}")
            return False
    
    def get_next_trading_day(self, date: str, n: int = 1) -> str:
        """
        获取第n个交易日
        
        Args:
            date: 起始日期（YYYYMMDD）
            n: 第n个交易日（正数向后，负数向前）
            
        Returns:
            str: 交易日日期（YYYYMMDD）
        """
        try:
            # 获取扩展的交易日历
            start_dt = pd.to_datetime(date) - timedelta(days=abs(n) * 10)
            end_dt = pd.to_datetime(date) + timedelta(days=abs(n) * 10)
            
            calendar = self.get_trading_calendar(
                start_dt.strftime('%Y%m%d'), 
                end_dt.strftime('%Y%m%d')
            )
            
            if calendar.empty:
                return date
            
            # 找到起始日期的位置
            date_dt = pd.to_datetime(date)
            calendar_dates = calendar['trade_date'].tolist()
            
            try:
                current_idx = calendar_dates.index(date_dt)
            except ValueError:
                # 如果起始日期不是交易日，找到最近的交易日
                future_dates = calendar[calendar['trade_date'] > date_dt]
                if not future_dates.empty:
                    current_idx = calendar_dates.index(future_dates.iloc[0]['trade_date'])
                else:
                    return date
            
            # 计算目标位置
            target_idx = current_idx + n
            
            if 0 <= target_idx < len(calendar_dates):
                return calendar_dates[target_idx].strftime('%Y%m%d')
            else:
                return date
                
        except Exception as e:
            logger.error(f"获取下一个交易日失败: {date}, n={n}, 错误: {e}")
            return date
    
    def format_data_for_prediction(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """
        格式化数据用于预测
        
        Args:
            stock_data: 原始股票数据
            
        Returns:
            pandas.DataFrame: 格式化后的数据
        """
        if stock_data.empty:
            return pd.DataFrame()
        
        # 复制数据避免修改原始数据
        formatted_data = stock_data.copy()
        
        # 确保必要的列存在
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in formatted_data.columns:
                logger.error(f"缺少必要列: {col}")
                return pd.DataFrame()
        
        # 设置日期为索引
        formatted_data = formatted_data.set_index('date')
        
        # 确保数据按日期排序
        formatted_data = formatted_data.sort_index()
        
        # 选择用于预测的列
        prediction_columns = ['open', 'high', 'low', 'close', 'volume']
        formatted_data = formatted_data[prediction_columns]
        
        # 处理缺失值
        formatted_data = formatted_data.fillna(method='ffill').fillna(method='bfill')
        
        return formatted_data

# 全局数据提供器实例
_data_provider = None

def get_data_provider() -> AkshareDataProvider:
    """获取全局数据提供器实例"""
    global _data_provider
    if _data_provider is None:
        _data_provider = AkshareDataProvider()
    return _data_provider

if __name__ == "__main__":
    # 测试数据提供器
    provider = AkshareDataProvider()
    
    # 测试股票数据获取
    print("测试股票数据获取...")
    data = provider.get_stock_data('000001', 'daily', '20240101', '20241026')
    print(f"数据形状: {data.shape}")
    print(data.head())
    
    # 测试股票搜索
    print("\n测试股票搜索...")
    results = provider.search_stock('平安')
    print(f"搜索结果数量: {len(results)}")
    for stock in results[:3]:
        print(f"{stock['symbol']} - {stock['name']}")
    
    # 测试交易日历
    print("\n测试交易日历...")
    calendar = provider.get_trading_calendar('20240101', '20240131')
    print(f"交易日数量: {len(calendar)}")
    
    # 测试交易日判断
    print("\n测试交易日判断...")
    is_trading = provider.is_trading_day('20241028')
    print(f"2024-10-28是交易日: {is_trading}")
    
    # 测试下一个交易日
    print("\n测试下一个交易日...")
    next_day = provider.get_next_trading_day('20241026', 1)
    print(f"2024-10-26的下一个交易日: {next_day}")