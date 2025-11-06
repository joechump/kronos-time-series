import pandas as pd

def test_date_parsing():
    """测试不同日期格式的解析"""
    print("测试日期格式解析:")
    
    # 前端发送的日期格式
    frontend_formats = [
        '2023-01-01T00:00',  # 前端实际发送的格式
        '2023-01-01',        # 用户输入的格式
        '2023-01-01T00:00:00'  # 完整格式
    ]
    
    for date_str in frontend_formats:
        try:
            parsed = pd.to_datetime(date_str)
            print(f"  {date_str} -> {parsed} (类型: {type(parsed)})")
        except Exception as e:
            print(f"  {date_str} -> 错误: {e}")
    
    print("\n测试后端处理逻辑:")
    
    # 模拟后端处理
    test_dates = [
        '2023-01-01T00:00',
        '2023-01-01',
        'invalid-date',
        None
    ]
    
    for start_date in test_dates:
        print(f"\n处理日期: {start_date}")
        if start_date:
            try:
                start_dt = pd.to_datetime(start_date)
                print(f"  解析成功: {start_dt}")
            except Exception as e:
                print(f"  解析失败: {e}")
        else:
            print("  日期为空")

if __name__ == "__main__":
    test_date_parsing()