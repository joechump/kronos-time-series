import subprocess
import json

# 测试start_date为"null"字符串的情况
print("正在测试 start_date='null' 字符串的处理...")

# 准备测试数据
data = {
    "file_path": "stock_600519_live",  # 使用实时股票数据
    "lookback": 400,
    "pred_len": 120,
    "start_date": "null"  # 这应该被当作None处理，使用最新数据
}

# 将数据写入临时文件
with open('test_data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False)

try:
    # 使用curl发送请求到API
    cmd = [
        'curl', 
        '-X', 'POST', 
        '-H', 'Content-Type: application/json',
        '-d', '@test_data.json',
        'http://127.0.0.1:7070/api/predict'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd='C:\\kron\\webui')
    
    print(f"返回码: {result.returncode}")
    print(f"标准输出: {result.stdout}")
    if result.stderr:
        print(f"标准错误: {result.stderr}")
        
    if result.returncode == 0:
        print("✅ SUCCESS: start_date='null' 字符串测试完成")
    else:
        print("❌ ERROR: 请求失败")
        
except Exception as e:
    print(f"❌ ERROR: {str(e)}")
finally:
    # 清理临时文件
    import os
    if os.path.exists('test_data.json'):
        os.remove('test_data.json')