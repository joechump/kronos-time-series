import time
import json
import requests
from playwright.sync_api import sync_playwright

def capture_predict_requests():
    with sync_playwright() as p:
        # 启动浏览器
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        
        # 存储捕获的请求
        captured_requests = []
        
        # 监听请求
        def log_request(request):
            if '/api/predict' in request.url:
                print(f"📡 捕获到预测请求: {request.method} {request.url}")
                
                request_data = {
                    'url': request.url,
                    'method': request.method,
                    'headers': dict(request.headers),
                    'post_data': request.post_data
                }
                
                captured_requests.append(request_data)
                
                if request.post_data:
                    print(f"📦 请求数据: {request.post_data}")
                    
                    # 立即测试这个请求
                    try:
                        test_request_data = json.loads(request.post_data)
                        print("🔍 立即测试这个请求...")
                        response = requests.post(
                            'http://localhost:8080/api/predict',
                            json=test_request_data,
                            headers={'Content-Type': 'application/json'}
                        )
                        print(f"📥 测试响应状态: {response.status_code}")
                        if response.status_code != 200:
                            print(f"❌ 错误响应: {response.text}")
                        else:
                            print("✅ 请求成功！")
                    except Exception as e:
                        print(f"⚠️ 测试失败: {e}")
        
        # 监听响应
        def log_response(response):
            if '/api/predict' in response.url:
                print(f"📥 收到响应: {response.status} {response.url}")
                try:
                    # 尝试获取响应内容
                    response_text = response.text()
                    print(f"📄 响应内容预览: {response_text[:200]}...")
                except:
                    print("❌ 无法读取响应内容")
        
        page.on('request', log_request)
        page.on('response', log_response)
        
        # 导航到页面
        print("🌐 正在导航到 http://localhost:8080...")
        page.goto('http://localhost:8080', wait_until='networkidle')
        
        print("✅ 页面加载完成！")
        print("\n🔍 请手动在浏览器中执行以下操作：")
        print("1. 输入股票代码（如：600519）")
        print("2. 点击'加载股票数据'按钮")
        print("3. 等待模型加载完成")
        print("4. 点击'开始预测'按钮")
        print("\n📊 我将实时捕获所有预测请求和响应...")
        print("⏰ 监控将持续60秒...")
        
        # 等待用户操作
        time.sleep(60)
        
        print(f"\n📋 捕获到的请求总数: {len(captured_requests)}")
        
        # 保存捕获的数据
        with open('captured_predict_requests.json', 'w', encoding='utf-8') as f:
            json.dump(captured_requests, f, indent=2, ensure_ascii=False)
        
        print("💾 请求数据已保存到 captured_predict_requests.json")
        
        # 保持浏览器打开以便查看
        print("🔍 浏览器将保持打开状态，您可以继续测试...")
        input("按Enter键关闭浏览器...")
        
        browser.close()

if __name__ == "__main__":
    capture_predict_requests()