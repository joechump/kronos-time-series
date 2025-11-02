# 简单的日志文件检查脚本
try:
    # 尝试以UTF-8编码读取
    with open('kronos_app.log', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print("UTF-8编码读取成功")
        print(f"文件总行数: {len(lines)}")
        print("\n最后10行内容:")
        for line in lines[-10:]:
            print(repr(line))  # 显示原始字符串表示
except UnicodeDecodeError as e:
    print(f"UTF-8编码读取失败: {e}")
    try:
        # 尝试以GBK编码读取
        with open('kronos_app.log', 'r', encoding='gbk') as f:
            lines = f.readlines()
            print("GBK编码读取成功")
            print(f"文件总行数: {len(lines)}")
            print("\n最后10行内容:")
            for line in lines[-10:]:
                print(repr(line))  # 显示原始字符串表示
    except UnicodeDecodeError as e2:
        print(f"GBK编码读取也失败: {e2}")
        # 尝试以二进制模式读取
        with open('kronos_app.log', 'rb') as f:
            data = f.read()
            print("二进制读取成功")
            print(f"文件大小: {len(data)} 字节")
            print("\n最后100字节:")
            print(data[-100:])