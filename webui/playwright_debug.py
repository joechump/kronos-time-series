import time
import json
from playwright.sync_api import sync_playwright

def debug_predict_requests():
    with sync_playwright() as p:
        # 启动浏览器
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        
        # 存储捕获的请求
        captured_requests = []
        
        # 监听所有网络请求
        def log_request(request):
            url = request.url
            method = request.method
            
            # 捕获所有API请求
            if '/api/' in url:
                print(f"📡 捕获到API请求: {method} {url}")
                
                request_data = {
                    'url': url,
                    'method': method,
                    'headers': dict(request.headers),
                    'post_data': request.post_data
                }
                
                captured_requests.append(request_data)
                
                if request.post_data:
                    print(f"📦 请求数据: {request.post_data}")
                    
                    # 如果是预测请求，立即测试
                    if '/api/predict' in url:
                        print("🔍 检测到预测请求，立即测试...")
                        try:
                            import requests
                            test_data = json.loads(request.post_data)
                            
                            print("🚀 发送测试请求到后端...")
                            response = requests.post(
                                'http://localhost:8080/api/predict',
                                json=test_data,
                                headers={'Content-Type': 'application/json'},
                                timeout=30
                            )
                            
                            print(f"📥 测试响应状态: {response.status_code}")
                            if response.status_code == 400:
                                print(f"❌ 400错误详情: {response.text}")
                            elif response.status_code == 200:
                                print("✅ 请求成功！")
                                result = response.json()
                                if 'predictions' in result:
                                    print(f"📊 预测点数: {len(result['predictions'])}")
                            else:
                                print(f"⚠️ 其他状态: {response.text}")
                                
                        except Exception as e:
                            print(f"💥 测试失败: {e}")
        
        # 监听响应
        def log_response(response):
            url = response.url
            if '/api/' in url:
                print(f"📥 收到响应: {response.status} {url}")
                try:
                    # 尝试获取响应内容
                    response_text = response.text()
                    if response.status >= 400:
                        print(f"❌ 错误响应: {response_text[:500]}...")
                    else:
                        print(f"✅ 成功响应预览: {response_text[:200]}...")
                except:
                    print("❌ 无法读取响应内容")
        
        page.on('request', log_request)
        page.on('response', log_response)
        
        # 导航到Kronos主页面
        print("🌐 正在导航到 http://localhost:8080...")
        page.goto('http://localhost:8080', wait_until='networkidle')
        
        print("✅ 页面加载完成！")
        print("\n🔍 开始模拟用户操作...")
        
        # 等待页面完全加载
        time.sleep(3)
        
        # 1. 输入股票代码
        print("1️⃣ 输入股票代码: 600519")
        try:
            page.fill('#stock-code-input', '600519')
            print("   ✅ 股票代码输入完成")
        except Exception as e:
            print(f"   ❌ 输入股票代码失败: {e}")
        
        time.sleep(1)
        
        # 2. 点击加载股票数据按钮
        print("2️⃣ 点击'加载股票数据'按钮")
        try:
            page.click('#load-stock-data')
            print("   ✅ 点击完成，等待数据加载...")
            
            # 等待数据加载完成（最多30秒）
            for i in range(30):
                time.sleep(1)
                # 检查是否有数据加载完成的迹象
                try:
                    current_data = page.evaluate('window.currentDataFile || ""')
                    if current_data:
                        print(f"   📊 当前数据文件: {current_data}")
                        break
                except:
                    pass
                
                if i == 29:
                    print("   ⚠️ 数据加载超时")
        except Exception as e:
            print(f"   ❌ 点击加载按钮失败: {e}")
        
        time.sleep(2)
        
        # 3. 检查模型状态
        print("3️⃣ 检查模型加载状态")
        try:
            model_loaded = page.evaluate('window.modelLoaded || false')
            print(f"   📋 模型加载状态: {'已加载' if model_loaded else '未加载'}")
            
            if not model_loaded:
                print("   ⚠️ 模型未加载，尝试等待...")
                for i in range(10):
                    time.sleep(2)
                    model_loaded = page.evaluate('window.modelLoaded || false')
                    if model_loaded:
                        print("   ✅ 模型已加载")
                        break
                    if i == 9:
                        print("   ❌ 模型加载超时")
        except Exception as e:
            print(f"   ❌ 检查模型状态失败: {e}")
        
        time.sleep(1)
        
        # 4. 点击开始预测按钮
        print("4️⃣ 点击'开始预测'按钮")
        try:
            # 先检查按钮是否可用
            button_disabled = page.evaluate('document.querySelector("#prediction-button").disabled')
            print(f"   📋 预测按钮状态: {'禁用' if button_disabled else '启用'}")
            
            if not button_disabled:
                page.click('#prediction-button')
                print("   ✅ 点击预测按钮完成")
                
                # 等待预测请求发送（最多20秒）
                for i in range(20):
                    time.sleep(1)
                    # 检查是否有预测请求被捕获
                    if any('/api/predict' in req['url'] for req in captured_requests):
                        print("   📡 预测请求已捕获")
                        break
                    if i == 19:
                        print("   ⚠️ 预测请求等待超时")
            else:
                print("   ❌ 预测按钮被禁用，无法点击")
                
                # 尝试手动启用按钮
                print("   🔧 尝试手动设置按钮状态...")
                page.evaluate('''
                    document.querySelector("#prediction-button").disabled = false;
                    window.modelLoaded = true;
                    window.currentDataFile = "stock_600519_live";
                ''')
                
                time.sleep(1)
                
                # 再次尝试点击
                button_disabled = page.evaluate('document.querySelector("#prediction-button").disabled')
                if not button_disabled:
                    page.click('#prediction-button')
                    print("   ✅ 手动设置后点击成功")
                else:
                    print("   ❌ 手动设置后按钮仍被禁用")
                    
        except Exception as e:
            print(f"   ❌ 点击预测按钮失败: {e}")
        
        # 等待额外时间捕获可能的请求
        print("\n⏰ 等待额外请求捕获（10秒）...")
        time.sleep(10)
        
        print(f"\n📋 捕获到的请求总数: {len(captured_requests)}")
        
        # 保存捕获的数据
        with open('detailed_captured_requests.json', 'w', encoding='utf-8') as f:
            json.dump(captured_requests, f, indent=2, ensure_ascii=False)
        
        print("💾 详细请求数据已保存到 detailed_captured_requests.json")
        
        # 保持浏览器打开以便查看
        print("\n🔍 浏览器将保持打开状态，您可以继续分析...")
        input("按Enter键关闭浏览器...")
        
        browser.close()

if __name__ == "__main__":
    debug_predict_requests()