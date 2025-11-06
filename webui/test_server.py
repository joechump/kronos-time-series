import requests
import time

def test_server():
    """测试服务器是否在运行"""
    try:
        # 等待几秒钟让服务器启动
        time.sleep(5)
        
        # 尝试访问服务器
        response = requests.get('http://localhost:7070/api/system-info', timeout=10)
        if response.status_code == 200:
            print("✅ 服务器正在运行!")
            print(f"响应内容: {response.json()}")
            return True
        else:
            print(f"❌ 服务器返回状态码: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务器，请检查服务器是否启动")
        return False
    except requests.exceptions.Timeout:
        print("❌ 连接超时，请检查服务器是否启动")
        return False
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        return False

if __name__ == "__main__":
    test_server()