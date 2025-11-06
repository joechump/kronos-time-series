import asyncio
from playwright.async_api import async_playwright
import json

async def capture_predict_requests():
    async with async_playwright() as p:
        # 启动浏览器
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        
        # 监听所有网络请求
        requests_data = []
        
        def log_request(request):
            if '/api/predict' in request.url:
                request_data = {
                    'url': request.url,
                    'method': request.method,
                    'headers': dict(request.headers),
                    'post_data': request.post_data
                }
                requests_data.append(request_data)
                print(f"📡 捕获到预测请求: {request.method} {request.url}")
                if request.post_data:
                    print(f"📦 请求数据: {request.post_data}")
        
        # 监听请求
        page.on('request', log_request)
        
        # 监听响应
        async def log_response(response):
            if '/api/predict' in response.url:
                print(f"📥 收到响应: {response.status} {response.url}")
                try:
                    response_body = await response.text()
                    print(f"📄 响应内容: {response_body[:500]}...")
                except:
                    print("❌ 无法读取响应内容")
        
        page.on('response', log_response)
        
        # 导航到页面
        print("🌐 正在导航到 http://localhost:8080...")
        await page.goto('http://localhost:8080', wait_until='networkidle')
        
        print("✅ 页面加载完成，等待用户操作...")
        print("🔍 请手动在浏览器中执行以下操作：")
        print("1. 输入股票代码（如：600519）")
        print("2. 点击'加载股票数据'按钮")
        print("3. 等待模型加载完成")
        print("4. 点击'开始预测'按钮")
        print("\n📊 我将实时捕获所有预测请求和响应...")
        
        # 等待用户操作（30秒）
        await asyncio.sleep(30)
        
        print(f"\n📋 捕获到的请求总数: {len(requests_data)}")
        for i, req in enumerate(requests_data):
            print(f"\n📊 请求 #{i+1}:")
            print(f"   URL: {req['url']}")
            print(f"   方法: {req['method']}")
            if req['post_data']:
                print(f"   数据: {req['post_data']}")
        
        # 保存捕获的数据到文件
        with open('captured_requests.json', 'w', encoding='utf-8') as f:
            json.dump(requests_data, f, indent=2, ensure_ascii=False)
        
        print("\n💾 请求数据已保存到 captured_requests.json")
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(capture_predict_requests())