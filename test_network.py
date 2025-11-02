import requests
import socket

# 测试网络连接
def test_network_connection():
    print("测试网络连接...")
    
    # 测试DNS解析
    try:
        socket.gethostbyname('www.baidu.com')
        print("DNS解析正常")
    except Exception as e:
        print(f"DNS解析失败: {e}")
        return False
    
    # 测试HTTP连接
    try:
        response = requests.get('http://www.baidu.com', timeout=5)
        print(f"HTTP连接正常，状态码: {response.status_code}")
        return True
    except Exception as e:
        print(f"HTTP连接失败: {e}")
        return False

if __name__ == "__main__":
    test_network_connection()