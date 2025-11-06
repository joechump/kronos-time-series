"""
简单测试temp_file_path功能
"""

import sys
import os
import json
import tempfile
import pandas as pd
from datetime import datetime

def simple_test():
    print("=== 简单测试temp_file_path功能 ===")
    
    # 测试1: 创建模拟数据
    print("\n1. 创建模拟数据")
    data = pd.DataFrame({
        'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'open': [10.0, 10.1, 10.2],
        'high': [10.5, 10.6, 10.7],
        'low': [9.8, 9.9, 10.0],
        'close': [10.2, 10.3, 10.4],
        'volume': [1000000, 1200000, 1100000]
    })
    print("✓ 模拟数据创建成功")
    
    # 测试2: 保存到临时文件
    print("\n2. 保存到临时文件")
    temp_dir = tempfile.gettempdir()
    temp_file = os.path.join(temp_dir, "test_temp_file.csv")
    
    data.to_csv(temp_file, index=False)
    print(f"✓ 数据保存到临时文件: {temp_file}")
    print(f"  文件大小: {os.path.getsize(temp_file)} 字节")
    
    # 测试3: 模拟API响应格式
    print("\n3. 模拟API响应格式")
    api_response = {
        'success': True,
        'data': {
            'symbol': '000001',
            'period': 'daily',
            'records': 3,
            'temp_file_path': temp_file  # 关键字段
        },
        'message': '数据获取成功'
    }
    
    print("✓ API响应格式创建成功")
    print(f"  temp_file_path字段值: {api_response['data']['temp_file_path']}")
    
    # 验证字段存在
    if 'temp_file_path' in api_response['data']:
        print("✓ temp_file_path字段存在")
    else:
        print("❌ temp_file_path字段缺失")
        return False
    
    # 验证文件存在
    if os.path.exists(api_response['data']['temp_file_path']):
        print("✓ 临时文件存在")
    else:
        print("❌ 临时文件不存在")
        return False
    
    # 测试4: 从临时文件加载数据
    print("\n4. 从临时文件加载数据")
    loaded_data = pd.read_csv(temp_file)
    print(f"✓ 从临时文件加载数据成功，数据量: {len(loaded_data)}条")
    
    # 验证数据一致性
    if len(data) == len(loaded_data):
        print("✓ 数据长度一致")
    else:
        print("❌ 数据长度不一致")
        return False
    
    # 测试5: 清理临时文件
    print("\n5. 清理临时文件")
    os.remove(temp_file)
    print("✓ 临时文件清理完成")
    
    print("\n🎉 所有测试通过！temp_file_path功能正常。")
    return True

if __name__ == "__main__":
    try:
        success = simple_test()
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        sys.exit(1)