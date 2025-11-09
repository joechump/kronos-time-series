#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试开始日期验证逻辑
"""

import pandas as pd

def test_date_validation():
    """测试开始日期验证逻辑"""
    
    # 测试日期
    start_date = "2025-11-07"
    max_allowed_date = pd.to_datetime('2025-12-31')
    
    print("=== 开始日期验证测试 ===")
    print(f"测试开始日期: {start_date}")
    print(f"最大允许日期: {max_allowed_date}")
    
    # 转换开始日期
    start_dt = pd.to_datetime(start_date)
    print(f"转换后的开始日期: {start_dt}")
    
    # 比较日期
    print(f"开始日期 > 最大允许日期: {start_dt > max_allowed_date}")
    print(f"开始日期 < 最大允许日期: {start_dt < max_allowed_date}")
    print(f"开始日期 == 最大允许日期: {start_dt == max_allowed_date}")
    
    # 检查日期范围
    print(f"开始日期是否在允许范围内: {start_dt <= max_allowed_date}")
    
    # 测试其他日期
    test_dates = ["2025-11-07", "2025-12-31", "2026-01-01", "2024-12-31"]
    print("\n=== 多个日期测试 ===")
    for test_date in test_dates:
        test_dt = pd.to_datetime(test_date)
        is_valid = test_dt <= max_allowed_date
        print(f"{test_date}: {is_valid} (<= {max_allowed_date})")

if __name__ == "__main__":
    test_date_validation()