# 启动Kronos Web服务器
cd C:\kron\webui
python app.py --port 7070 --host 127.0.0.1 --debug
Write-Host "服务器已启动，请按任意键退出..."
$host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")