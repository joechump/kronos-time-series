#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
前端400错误测试脚本
使用正确的Chromium路径测试前端预测功能
"""

import time
import json
import os
from playwright.sync_api import sync_playwright

def test_frontend_predict():
    """测试前端预测功能"""
    
    print("🔍 开始测试前端预测功能...")
    
    # 可能的Chromium路径
    chromium_paths = [
        "C:\\Users\\Administrator\\AppData\\Local\\ms-playwright\\chromium-1187\\chrome-win\\chrome.exe",
        "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
        "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
    ]
    
    # 查找可用的Chromium路径
    chromium_path = None
    for path in chromium_paths:
        if os.path.exists(path):
            chromium_path = path
            print(f"✅ 找到Chromium路径: {path}")
            break
    
    if not chromium_path:
        print("❌ 未找到Chromium路径，尝试使用默认路径")
    
    with sync_playwright() as p:
        # 配置浏览器选项
        browser_options = {
            'headless': False,
            'timeout': 30000
        }
        
        # 如果找到路径，使用指定路径
        if chromium_path:
            browser_options['executable_path'] = chromium_path
        
        try:
            browser = p.chromium.launch(**browser_options)
            page = browser.new_page()
            page.set_viewport_size({"width": 1280, "height": 800})
            
            # 存储捕获的数据
            captured_data = {
                'requests': [],
                'responses': [],
                'console_logs': [],
                'errors': []
            }
            
            # 监听控制台日志
            def log_console(msg):
                log_entry = {
                    'type': msg.type,
                    'text': msg.text,
                    'timestamp': time.time()
                }
                captured_data['console_logs'].append(log_entry)
                print(f"📝 控制台[{msg.type}]: {msg.text}")
            
            # 监听页面错误
            def log_error(error):
                error_entry = {
                    'error': str(error),
                    'timestamp': time.time()
                }
                captured_data['errors'].append(error_entry)
                print(f"💥 页面错误: {error}")
            
            # 监听网络请求
            def log_request(request):
                if '/api/' in request.url:
                    request_data = {
                        'url': request.url,
                        'method': request.method,
                        'headers': dict(request.headers),
                        'post_data': request.post_data,
                        'timestamp': time.time()
                    }
                    captured_data['requests'].append(request_data)
                    
                    print(f"📡 API请求: {request.method} {request.url}")
                    if request.post_data:
                        print(f"   📦 请求数据: {request.post_data}")
            
            # 监听网络响应
            def log_response(response):
                if '/api/' in response.url:
                    response_data = {
                        'url': response.url,
                        'status': response.status,
                        'headers': dict(response.headers),
                        'timestamp': time.time()
                    }
                    
                    # 尝试获取响应内容
                    try:
                        response_text = response.text()
                        response_data['body'] = response_text[:500]  # 限制长度
                    except:
                        response_data['body'] = "无法读取响应内容"
                    
                    captured_data['responses'].append(response_data)
                    
                    status_icon = "✅" if response.status == 200 else "❌"
                    print(f"{status_icon} API响应: {response.status} {response.url}")
                    
                    if response.status >= 400:
                        print(f"   💥 错误详情: {response_data['body']}")
            
            # 设置监听器
            page.on('console', log_console)
            page.on('pageerror', log_error)
            page.on('request', log_request)
            page.on('response', log_response)
            
            # 导航到页面
            print("\n🌐 正在导航到 http://localhost:8080...")
            page.goto('http://localhost:8080', wait_until='networkidle')
            
            print("✅ 页面加载完成！")
            time.sleep(3)
            
            # 检查关键元素
            print("\n🔍 检查页面关键元素...")
            
            elements = [
                '#stock-code-input',
                '#load-data-btn', 
                '#predict-btn'
            ]
            
            for selector in elements:
                element = page.query_selector(selector)
                if element:
                    print(f"   ✅ 元素存在: {selector}")
                    print(f"     可见性: {element.is_visible()}, 启用状态: {element.is_enabled()}")
                else:
                    print(f"   ❌ 元素不存在: {selector}")
            
            # 模拟用户操作
            print("\n🔄 开始模拟用户操作...")
            
            # 1. 输入股票代码
            print("1️⃣ 输入股票代码: 600519")
            page.fill('#stock-code-input', '600519')
            time.sleep(1)
            
            # 2. 点击加载数据按钮
            print("2️⃣ 点击'加载股票数据'按钮")
            page.click('#load-data-btn')
            
            # 等待数据加载
            print("   ⏳ 等待数据加载...")
            time.sleep(10)
            
            # 3. 检查数据加载状态
            print("3️⃣ 检查数据加载状态")
            try:
                data_file = page.evaluate('window.currentDataFile || ""')
                model_loaded = page.evaluate('window.modelLoaded || false')
                
                print(f"   📊 数据文件: {data_file}")
                print(f"   🤖 模型加载: {model_loaded}")
                
                if not data_file:
                    print("   ⚠️ 数据文件未设置，尝试手动设置")
                    page.evaluate('window.currentDataFile = "stock_600519_live"')
                    
                if not model_loaded:
                    print("   ⚠️ 模型未加载，等待10秒...")
                    time.sleep(10)
                    model_loaded = page.evaluate('window.modelLoaded || false')
                    print(f"   🤖 重新检查模型状态: {model_loaded}")
                    
            except Exception as e:
                print(f"   ❌ 检查状态失败: {e}")
            
            # 4. 点击预测按钮
            print("4️⃣ 点击'开始预测'按钮")
            
            # 检查按钮状态
            predict_button = page.query_selector('#predict-btn')
            if predict_button:
                is_disabled = predict_button.get_attribute('disabled')
                print(f"   📋 预测按钮禁用状态: {is_disabled}")
                
                if is_disabled:
                    print("   🔧 按钮被禁用，尝试手动启用")
                    page.evaluate('''
                        window.currentDataFile = "stock_600519_live";
                        window.modelLoaded = true;
                        document.querySelector("#predict-btn").disabled = false;
                    ''')
                    time.sleep(2)
                    
                    # 重新检查
                    is_disabled = page.query_selector('#predict-btn').get_attribute('disabled')
                    print(f"   📋 手动设置后禁用状态: {is_disabled}")
                
                if not is_disabled:
                    print("   🖱️ 点击预测按钮...")
                    page.click('#predict-btn')
                    
                    # 等待预测请求
                    print("   ⏳ 等待预测请求...")
                    time.sleep(10)
                    
                    # 检查是否有预测请求
                    predict_requests = [req for req in captured_data['requests'] 
                                      if '/api/predict' in req['url']]
                    
                    if predict_requests:
                        print("   ✅ 预测请求已捕获")
                        for req in predict_requests:
                            print(f"     请求数据: {req.get('post_data', 'N/A')}")
                    else:
                        print("   ❌ 未捕获到预测请求")
                else:
                    print("   ❌ 预测按钮仍被禁用，无法点击")
            else:
                print("   ❌ 预测按钮未找到")
            
            # 等待额外时间捕获请求
            print("\n⏰ 等待额外请求捕获（10秒）...")
            time.sleep(10)
            
            # 分析结果
            print("\n📊 测试结果分析:")
            print(f"   捕获的API请求数: {len(captured_data['requests'])}")
            print(f"   捕获的API响应数: {len(captured_data['responses'])}")
            print(f"   控制台日志数: {len(captured_data['console_logs'])}")
            print(f"   页面错误数: {len(captured_data['errors'])}")
            
            # 检查400错误
            error_responses = [resp for resp in captured_data['responses'] 
                             if resp['status'] >= 400]
            
            if error_responses:
                print("\n❌ 发现错误响应:")
                for error in error_responses:
                    print(f"   {error['status']} {error['url']}")
                    print(f"   错误详情: {error['body']}")
            else:
                print("\n✅ 未发现错误响应")
            
            # 保存调试数据
            with open('frontend_test_results.json', 'w', encoding='utf-8') as f:
                json.dump(captured_data, f, indent=2, ensure_ascii=False)
            
            print("\n💾 调试数据已保存到 frontend_test_results.json")
            
            # 保持浏览器打开以便检查
            print("\n🖥️ 浏览器保持打开状态，请检查页面...")
            input("按Enter键关闭浏览器...")
            
            browser.close()
            
        except Exception as e:
            print(f"💥 浏览器操作失败: {e}")
            import traceback
            traceback.print_exc()

def main():
    """主函数"""
    
    print("🔍 Kronos 前端预测功能测试")
    print("=" * 50)
    
    # 检查后端服务
    try:
        import requests
        response = requests.get('http://localhost:8080/', timeout=5)
        print("✅ 后端服务运行正常")
    except:
        print("❌ 后端服务未运行，请先启动webui服务")
        return
    
    # 运行前端测试
    test_frontend_predict()
    
    print("\n🎯 测试完成！")

if __name__ == "__main__":
    main()