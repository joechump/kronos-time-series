import time
from playwright.sync_api import sync_playwright

def check_page_structure():
    print("🔍 检查页面结构...")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        
        # 导航到页面
        page.goto('http://localhost:8080', wait_until='networkidle')
        time.sleep(3)
        
        print("\n📋 检查所有按钮元素...")
        
        # 查找所有按钮元素
        buttons = page.query_selector_all('button')
        print(f"📊 页面中共有 {len(buttons)} 个按钮")
        
        for i, button in enumerate(buttons):
            button_id = button.get_attribute('id') or '无ID'
            button_text = button.inner_text().strip() or '无文本'
            button_class = button.get_attribute('class') or '无类名'
            
            print(f"\n🔘 按钮 {i+1}:")
            print(f"   ID: {button_id}")
            print(f"   文本: {button_text}")
            print(f"   类名: {button_class}")
        
        print("\n🔍 检查输入框元素...")
        
        # 查找所有输入框
        inputs = page.query_selector_all('input')
        print(f"📊 页面中共有 {len(inputs)} 个输入框")
        
        for i, input_elem in enumerate(inputs):
            input_id = input_elem.get_attribute('id') or '无ID'
            input_type = input_elem.get_attribute('type') or '无类型'
            input_placeholder = input_elem.get_attribute('placeholder') or '无占位符'
            
            print(f"\n📝 输入框 {i+1}:")
            print(f"   ID: {input_id}")
            print(f"   类型: {input_type}")
            print(f"   占位符: {input_placeholder}")
        
        print("\n🔍 检查页面HTML结构...")
        
        # 获取页面HTML结构
        html_content = page.content()
        
        # 查找包含"股票"相关的内容
        stock_related = []
        lines = html_content.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['stock', '股票', '预测', 'predict', '加载', 'load']):
                stock_related.append(line.strip())
        
        print("\n📋 股票相关HTML片段:")
        for i, line in enumerate(stock_related[:20]):  # 只显示前20行
            print(f"{i+1}: {line}")
        
        # 保持浏览器打开
        input("\n按Enter键关闭浏览器...")
        browser.close()

if __name__ == "__main__":
    check_page_structure()