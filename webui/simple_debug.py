import time
import json
import requests
from playwright.sync_api import sync_playwright

def simple_debug():
    print("🔍 开始简单调试模式...")
    
    # 先直接测试后端API
    print("\n1️⃣ 直接测试后端API...")
    
    # 测试数据
    test_data = {
        "stock_code": "600519",
        "model_name": "kronos-small",
        "data_file": "stock_600519_live",
        "prediction_days": 30
    }
    
    print(f"📤 发送测试数据: {json.dumps(test_data, ensure_ascii=False)}")
    
    try:
        response = requests.post(
            'http://localhost:8080/api/predict',
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        print(f"📥 响应状态: {response.status_code}")
        if response.status_code == 400:
            print(f"❌ 400错误详情: {response.text}")
        elif response.status_code == 200:
            print("✅ 直接API测试成功！")
            result = response.json()
            if 'predictions' in result:
                print(f"📊 预测点数: {len(result['predictions'])}")
        else:
            print(f"⚠️ 其他状态: {response.text}")
            
    except Exception as e:
        print(f"💥 直接API测试失败: {e}")
    
    print("\n2️⃣ 使用Playwright捕获前端请求...")
    
    captured_requests = []
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        
        # 监听请求
        def capture_request(request):
            url = request.url
            if '/api/predict' in url:
                print(f"🎯 捕获到预测请求: {request.method} {url}")
                
                request_info = {
                    'url': url,
                    'method': request.method,
                    'headers': dict(request.headers),
                    'post_data': request.post_data,
                    'timestamp': time.time()
                }
                
                captured_requests.append(request_info)
                
                if request.post_data:
                    print(f"📦 请求数据: {request.post_data}")
                    
                    # 立即测试这个请求
                    try:
                        test_response = requests.post(
                            'http://localhost:8080/api/predict',
                            json=json.loads(request.post_data),
                            headers={'Content-Type': 'application/json'},
                            timeout=30
                        )
                        
                        print(f"🔬 测试结果: {test_response.status_code}")
                        if test_response.status_code == 400:
                            print(f"❌ 400错误: {test_response.text}")
                        
                    except Exception as e:
                        print(f"💥 测试失败: {e}")
        
        page.on('request', capture_request)
        
        # 导航到页面
        print("🌐 导航到 http://localhost:8080...")
        page.goto('http://localhost:8080', wait_until='networkidle')
        
        print("✅ 页面加载完成")
        
        # 等待页面初始化
        time.sleep(3)
        
        # 检查页面元素
        print("\n3️⃣ 检查页面元素状态...")
        
        try:
            # 检查股票代码输入框
            stock_input = page.query_selector('#stock-code-input')
            if stock_input:
                print("✅ 找到股票代码输入框")
                stock_input.fill('600519')
                print("✅ 输入股票代码: 600519")
            else:
                print("❌ 未找到股票代码输入框")
                
            # 检查加载数据按钮
            load_button = page.query_selector('#load-stock-data')
            if load_button:
                print("✅ 找到加载数据按钮")
                load_button.click()
                print("✅ 点击加载数据按钮")
            else:
                print("❌ 未找到加载数据按钮")
                
            # 等待数据加载
            time.sleep(5)
            
            # 检查预测按钮
            predict_button = page.query_selector('#prediction-button')
            if predict_button:
                disabled = predict_button.get_attribute('disabled')
                print(f"📋 预测按钮状态: {'禁用' if disabled else '启用'}")
                
                if not disabled:
                    predict_button.click()
                    print("✅ 点击预测按钮")
                else:
                    print("❌ 预测按钮被禁用")
                    
                    # 尝试手动启用
                    page.evaluate('''
                        if (document.querySelector("#prediction-button")) {
                            document.querySelector("#prediction-button").disabled = false;
                            console.log("手动启用预测按钮");
                        }
                    ''')
                    
                    time.sleep(1)
                    
                    predict_button = page.query_selector('#prediction-button')
                    disabled = predict_button.get_attribute('disabled')
                    if not disabled:
                        predict_button.click()
                        print("✅ 手动启用后点击预测按钮")
                    else:
                        print("❌ 手动启用失败")
                        
            else:
                print("❌ 未找到预测按钮")
                
        except Exception as e:
            print(f"💥 页面操作失败: {e}")
        
        # 等待请求捕获
        print("\n⏰ 等待请求捕获（15秒）...")
        time.sleep(15)
        
        print(f"\n📋 总共捕获到 {len(captured_requests)} 个预测请求")
        
        # 保存捕获的数据
        if captured_requests:
            with open('simple_captured_requests.json', 'w', encoding='utf-8') as f:
                json.dump(captured_requests, f, indent=2, ensure_ascii=False)
            print("💾 请求数据已保存到 simple_captured_requests.json")
        
        # 保持浏览器打开
        print("\n🔍 浏览器保持打开，您可以继续分析...")
        input("按Enter键关闭浏览器...")
        
        browser.close()

if __name__ == "__main__":
    simple_debug()