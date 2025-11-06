#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用正确Chromium路径调试400错误
专门针对chromium-1187目录进行配置
"""

import time
import json
import os
import sys
from playwright.sync_api import sync_playwright

def find_chromium_path():
    """查找可用的Chromium路径"""
    possible_paths = [
        # 标准Playwright安装路径
        "C:\\Users\\Administrator\\AppData\\Local\\ms-playwright\\chromium-1187\\chrome-win\\chrome.exe",
        "C:\\Users\\Administrator\\AppData\\Local\\ms-playwright\\chromium-1187\\chrome-win\\chrome",
        # 其他可能的路径
        "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
        "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ 找到Chromium路径: {path}")
            return path
    
    print("❌ 未找到Chromium路径，将使用默认路径")
    return None

def debug_400_error():
    """调试400错误的主函数"""
    
    # 查找Chromium路径
    chromium_path = find_chromium_path()
    
    with sync_playwright() as p:
        # 配置浏览器启动选项
        browser_options = {
            'headless': False,
            'timeout': 30000
        }
        
        # 如果找到路径，使用指定路径
        if chromium_path:
            browser_options['executable_path'] = chromium_path
            print(f"🚀 使用指定Chromium路径启动浏览器...")
        else:
            print("🚀 使用默认Chromium路径启动浏览器...")
        
        try:
            browser = p.chromium.launch(**browser_options)
            page = browser.new_page()
            
            # 设置视口大小
            page.set_viewport_size({"width": 1280, "height": 800})
            
            # 存储捕获的请求和错误
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
                        response_data['body'] = response_text[:1000]  # 限制长度
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
            
            # 获取页面HTML结构用于调试
            html_content = page.content()
            with open('page_debug_content.html', 'w', encoding='utf-8') as f:
                f.write(html_content)
            print("📄 页面HTML已保存到 page_debug_content.html")
            
            # 检查关键元素是否存在
            print("\n🔍 检查页面关键元素...")
            
            elements_to_check = [
                '#stock-code-input',
                '#load-data-btn', 
                '#predict-btn',
                '#model-status',
                '#data-status'
            ]
            
            for selector in elements_to_check:
                try:
                    element = page.query_selector(selector)
                    if element:
                        print(f"   ✅ 元素存在: {selector}")
                        # 检查元素状态
                        is_visible = element.is_visible()
                        is_enabled = element.is_enabled()
                        print(f"     可见性: {is_visible}, 启用状态: {is_enabled}")
                    else:
                        print(f"   ❌ 元素不存在: {selector}")
                except Exception as e:
                    print(f"   ⚠️ 检查元素失败 {selector}: {e}")
            
            # 模拟用户操作流程
            print("\n🔄 开始模拟用户操作流程...")
            
            # 1. 输入股票代码
            print("1️⃣ 输入股票代码: 600519")
            try:
                page.fill('#stock-code-input', '600519')
                print("   ✅ 股票代码输入完成")
                time.sleep(1)
            except Exception as e:
                print(f"   ❌ 输入股票代码失败: {e}")
            
            # 2. 点击加载数据按钮
            print("2️⃣ 点击'加载股票数据'按钮")
            try:
                # 检查按钮状态
                button = page.query_selector('#load-data-btn')
                if button and button.is_enabled():
                    page.click('#load-data-btn')
                    print("   ✅ 点击加载数据按钮")
                    
                    # 等待数据加载（最多30秒）
                    print("   ⏳ 等待数据加载...")
                    for i in range(30):
                        time.sleep(1)
                        
                        # 检查数据状态
                        try:
                            data_status = page.evaluate('''
                                window.currentDataFile || ""
                            ''')
                            if data_status:
                                print(f"   📊 数据文件已设置: {data_status}")
                                break
                        except:
                            pass
                        
                        if i == 29:
                            print("   ⚠️ 数据加载超时")
                else:
                    print("   ❌ 加载数据按钮不可用")
                    
            except Exception as e:
                print(f"   ❌ 点击加载数据按钮失败: {e}")
            
            time.sleep(2)
            
            # 3. 检查模型状态
            print("3️⃣ 检查模型状态")
            try:
                model_status = page.evaluate('''
                    window.modelLoaded || false
                ''')
                print(f"   📋 模型加载状态: {'已加载' if model_status else '未加载'}")
                
                if not model_status:
                    print("   ⚠️ 模型未加载，等待10秒...")
                    time.sleep(10)
                    
                    # 重新检查
                    model_status = page.evaluate('''
                        window.modelLoaded || false
                    ''')
                    print(f"   📋 重新检查模型状态: {'已加载' if model_status else '未加载'}")
                    
            except Exception as e:
                print(f"   ❌ 检查模型状态失败: {e}")
            
            # 4. 点击预测按钮
            print("4️⃣ 点击'开始预测'按钮")
            try:
                # 检查预测按钮状态
                predict_button = page.query_selector('#predict-btn')
                if predict_button:
                    is_disabled = predict_button.get_attribute('disabled')
                    print(f"   📋 预测按钮禁用状态: {is_disabled}")
                    
                    if not is_disabled:
                        # 在点击前保存当前状态
                        current_data_file = page.evaluate('window.currentDataFile || ""')
                        model_loaded = page.evaluate('window.modelLoaded || false')
                        
                        print(f"   📊 点击前状态 - 数据文件: {current_data_file}, 模型加载: {model_loaded}")
                        
                        # 点击预测按钮
                        page.click('#predict-btn')
                        print("   ✅ 点击预测按钮完成")
                        
                        # 等待预测请求（最多20秒）
                        print("   ⏳ 等待预测请求...")
                        for i in range(20):
                            time.sleep(1)
                            
                            # 检查是否有预测请求
                            predict_requests = [req for req in captured_data['requests'] 
                                              if '/api/predict' in req['url']]
                            if predict_requests:
                                print("   📡 预测请求已捕获")
                                break
                            
                            if i == 19:
                                print("   ⚠️ 预测请求等待超时")
                    else:
                        print("   ❌ 预测按钮被禁用，无法点击")
                        
                        # 尝试手动启用
                        print("   🔧 尝试手动设置状态...")
                        page.evaluate('''
                            window.currentDataFile = "stock_600519_live";
                            window.modelLoaded = true;
                            document.querySelector("#predict-btn").disabled = false;
                        ''')
                        
                        time.sleep(2)
                        
                        # 重新检查并点击
                        is_disabled = page.query_selector('#predict-btn').get_attribute('disabled')
                        if not is_disabled:
                            page.click('#predict-btn')
                            print("   ✅ 手动设置后点击成功")
                        else:
                            print("   ❌ 手动设置后按钮仍被禁用")
                else:
                    print("   ❌ 预测按钮未找到")
                    
            except Exception as e:
                print(f"   ❌ 点击预测按钮失败: {e}")
            
            # 等待额外时间捕获可能的请求
            print("\n⏰ 等待额外请求捕获（15秒）...")
            time.sleep(15)
            
            # 保存所有捕获的数据
            print("\n💾 保存调试数据...")
            with open('detailed_400_debug_data.json', 'w', encoding='utf-8') as f:
                json.dump(captured_data, f, indent=2, ensure_ascii=False)
            
            print("✅ 调试数据已保存到 detailed_400_debug_data.json")
            
            # 分析结果
            print("\n📊 调试结果分析:")
            print(f"   捕获的API请求数: {len(captured_data['requests'])}")
            print(f"   捕获的API响应数: {len(captured_data['responses'])}")
            print(f"   控制台日志数: {len(captured_data['console_logs'])}")
            print(f"   页面错误数: {len(captured_data['errors'])}")
            
            # 检查是否有400错误
            error_responses = [resp for resp in captured_data['responses'] 
                             if resp['status'] >= 400]
            if error_responses:
                print("\n❌ 发现错误响应:")
                for error in error_responses:
                    print(f"   {error['status']} {error['url']}")
                    print(f"   错误详情: {error['body']}")
            else:
                print("\n✅ 未发现错误响应")
            
            # 关闭浏览器
            browser.close()
            
        except Exception as e:
            print(f"💥 浏览器操作失败: {e}")
            import traceback
            traceback.print_exc()

def main():
    """主函数"""
    print("🔍 使用Chromium-1187调试400错误")
    print("=" * 50)
    
    # 检查后端服务是否运行
    try:
        import requests
        response = requests.get('http://localhost:8080/', timeout=5)
        print("✅ 后端服务运行正常")
    except:
        print("❌ 后端服务未运行，请先启动webui服务")
        return
    
    # 运行调试
    debug_400_error()
    
    print("\n🎯 调试完成！")
    print("💡 请查看生成的调试文件分析具体问题")

if __name__ == "__main__":
    main()