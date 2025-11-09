// 前端预测请求调试脚本
// 在浏览器控制台中运行此脚本，捕获实际的预测请求参数

function debugPredictionRequest() {
    console.log('🔍 开始调试预测请求...');
    
    // 检查当前状态
    const stockCode = document.getElementById('stock-code-input')?.value?.trim() || '';
    const modelLoaded = window.modelLoaded || false;
    const currentDataFile = window.currentDataFile || '';
    
    console.log('📊 当前状态:');
    console.log('   股票代码:', stockCode);
    console.log('   模型加载状态:', modelLoaded);
    console.log('   当前数据文件:', currentDataFile);
    
    // 获取预测参数
    function getParamValue(paramId, defaultValue) {
        const element = document.getElementById(paramId);
        if (!element) {
            console.log(`❌ 参数元素未找到: ${paramId}`);
            return defaultValue;
        }
        const value = element.value;
        return (value !== null && value !== undefined && value !== '') ? value : defaultValue;
    }
    
    const lookback = parseInt(getParamValue('lookback', '100'));
    const predLen = parseInt(getParamValue('pred-len', '30'));
    const startDate = '使用默认日期计算';
    const temperature = parseFloat(getParamValue('temperature', '1.3'));
    const topP = parseFloat(getParamValue('top-p', '0.98'));
    const sampleCount = parseInt(getParamValue('sample-count', '2'));
    
    console.log('📋 预测参数:');
    console.log('   lookback:', lookback);
    console.log('   pred_len:', predLen);
    console.log('   start_date:', startDate);
    console.log('   temperature:', temperature);
    console.log('   top_p:', topP);
    console.log('   sample_count:', sampleCount);
    
    // 构建预测请求参数
    const predictionParams = {
        file_path: currentDataFile,
        lookback: lookback,
        pred_len: predLen,
        start_date: startDate,
        temperature: temperature,
        top_p: topP,
        sample_count: sampleCount
    };
    
    console.log('🚀 即将发送的预测请求参数:');
    console.log(JSON.stringify(predictionParams, null, 2));
    
    // 模拟发送请求（不实际发送，只显示参数）
    console.log('📤 请求URL: http://localhost:8080/api/predict');
    console.log('📤 请求方法: POST');
    console.log('📤 请求头: Content-Type: application/json');
    
    return predictionParams;
}

// 运行调试函数
const requestParams = debugPredictionRequest();

// 提供复制到剪贴板的功能
function copyRequestToClipboard() {
    const requestData = {
        url: 'http://localhost:8080/api/predict',
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        data: requestParams
    };
    
    const textToCopy = JSON.stringify(requestData, null, 2);
    
    navigator.clipboard.writeText(textToCopy).then(() => {
        console.log('✅ 请求参数已复制到剪贴板');
        console.log('📋 复制的内容:');
        console.log(textToCopy);
    }).catch(err => {
        console.error('❌ 复制失败:', err);
    });
}

console.log('💡 使用方法:');
console.log('1. 在浏览器中打开开发者工具 (F12)');
console.log('2. 切换到控制台 (Console) 标签页');
console.log('3. 粘贴并运行此脚本');
console.log('4. 查看实际的预测请求参数');
console.log('5. 运行 copyRequestToClipboard() 复制请求参数');

// 导出函数供外部调用
window.debugPredictionRequest = debugPredictionRequest;
window.copyRequestToClipboard = copyRequestToClipboard;