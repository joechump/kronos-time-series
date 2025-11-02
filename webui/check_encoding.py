import chardet

# 检查日志文件的编码
with open('kronos_app.log', 'rb') as f:
    raw_data = f.read()
    encoding = chardet.detect(raw_data)
    print(f"文件编码: {encoding['encoding']}")
    print(f"置信度: {encoding['confidence']}")
    
    # 尝试解码最后几行
    try:
        lines = raw_data.decode(encoding['encoding']).split('\n')
        print("\n最后10行内容:")
        for line in lines[-10:]:
            print(line)
    except Exception as e:
        print(f"解码失败: {e}")