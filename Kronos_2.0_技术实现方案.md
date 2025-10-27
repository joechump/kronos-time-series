# Kronos 2.0 技术实现方案

## 📋 项目概述

**当前版本**: Kronos 1.0  
**目标版本**: Kronos 2.0  
**技术栈**: Flask + Python + JavaScript + Plotly.js  
**数据源**: akshare (替代本地CSV)  
**界面语言**: 中文  

## 🎯 核心功能改进目标

### 1. 智能模型自动加载系统
- 自动检测系统资源并选择最优模型
- 无需用户手动选择模型和设备
- 提供高级用户手动配置选项

### 2. akshare数据源集成
- 实时获取股票历史数据
- 支持A股、港股、美股等市场
- 数据缓存和更新机制

### 3. 中文界面优化
- 完整的中文界面和操作提示
- 符合中文用户习惯的布局
- 简化的操作流程

### 4. 预测图表美观化
- 现代化的图表样式设计
- 增强的交互功能
- 响应式布局适配

### 5. 交易日历集成
- 基于实际交易日历的预测
- 跳过节假日和周末
- 智能时间轴计算

## 🔧 技术架构设计

### 后端架构改进

#### 2.1 自动模型加载模块
```python
# auto_model_loader.py
import psutil
import torch

class AutoModelLoader:
    def __init__(self):
        self.available_models = {
            'kronos-mini': {
                'name': 'Kronos-mini',
                'model_id': 'NeoQuasar/Kronos-mini',
                'memory_required': 2,  # GB
                'priority': 1
            },
            'kronos-small': {
                'name': 'Kronos-small', 
                'model_id': 'NeoQuasar/Kronos-small',
                'memory_required': 4,  # GB
                'priority': 2
            },
            'kronos-base': {
                'name': 'Kronos-base',
                'model_id': 'NeoQuasar/Kronos-base', 
                'memory_required': 8,  # GB
                'priority': 3
            }
        }
    
    def detect_system_resources(self):
        """检测系统资源"""
        memory_gb = psutil.virtual_memory().total / (1024**3)
        gpu_available = torch.cuda.is_available()
        mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        
        return {
            'memory_gb': memory_gb,
            'gpu_available': gpu_available,
            'mps_available': mps_available
        }
    
    def select_optimal_model(self):
        """选择最优模型"""
        resources = self.detect_system_resources()
        
        # 根据内存选择模型
        for model_key, model_info in sorted(
            self.available_models.items(), 
            key=lambda x: x[1]['priority']
        ):
            if resources['memory_gb'] >= model_info['memory_required']:
                return model_key, model_info
        
        # 如果内存不足，返回最小模型
        return 'kronos-mini', self.available_models['kronos-mini']
    
    def select_optimal_device(self):
        """选择最优设备"""
        resources = self.detect_system_resources()
        
        if resources['gpu_available']:
            return 'cuda'
        elif resources['mps_available']:
            return 'mps'
        else:
            return 'cpu'
```

#### 2.2 akshare数据源模块
```python
# akshare_data_provider.py
import akshare as ak
import pandas as pd
import datetime
from typing import Optional, Dict, Any

class AkshareDataProvider:
    def __init__(self):
        self.cache = {}
        self.cache_duration = 3600  # 1小时缓存
    
    def get_stock_data(self, symbol: str, period: str = "daily", 
                      start_date: Optional[str] = None, 
                      end_date: Optional[str] = None) -> pd.DataFrame:
        """获取股票历史数据"""
        
        # 生成缓存键
        cache_key = f"{symbol}_{period}_{start_date}_{end_date}"
        
        # 检查缓存
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if datetime.datetime.now().timestamp() - timestamp < self.cache_duration:
                return cached_data.copy()
        
        try:
            # 根据市场类型选择不同的akshare接口
            if symbol.startswith(('sh', 'sz')):
                # A股数据
                df = ak.stock_zh_a_hist(symbol=symbol, period=period, 
                                      start_date=start_date, end_date=end_date)
                # 重命名列以匹配Kronos格式
                df = df.rename(columns={
                    '日期': 'timestamps',
                    '开盘': 'open',
                    '最高': 'high', 
                    '最低': 'low',
                    '收盘': 'close',
                    '成交量': 'volume'
                })
            elif symbol.startswith(('hk', '0')):
                # 港股数据
                df = ak.stock_hk_hist(symbol=symbol, period=period,
                                    start_date=start_date, end_date=end_date)
                df = df.rename(columns={
                    '日期': 'timestamps',
                    '开盘': 'open',
                    '最高': 'high',
                    '最低': 'low', 
                    '收盘': 'close',
                    '成交量': 'volume'
                })
            else:
                # 美股数据
                df = ak.stock_us_hist(symbol=symbol, period=period,
                                    start_date=start_date, end_date=end_date)
                df = df.rename(columns={
                    '日期': 'timestamps',
                    '开盘': 'open',
                    '最高': 'high',
                    '最低': 'low',
                    '收盘': 'close', 
                    '成交量': 'volume'
                })
            
            # 确保时间戳格式正确
            df['timestamps'] = pd.to_datetime(df['timestamps'])
            
            # 确保数值列格式正确
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 缓存数据
            self.cache[cache_key] = (df.copy(), datetime.datetime.now().timestamp())
            
            return df
            
        except Exception as e:
            raise Exception(f"获取股票数据失败: {str(e)}")
    
    def search_stock(self, keyword: str) -> list:
        """搜索股票"""
        try:
            # A股搜索
            if keyword.isdigit() or any(keyword.startswith(prefix) for prefix in ['sh', 'sz']):
                df = ak.stock_info_a_code_name()
                results = df[df['code'].str.contains(keyword) | df['name'].str.contains(keyword)]
                return results.to_dict('records')
            
            # 港股搜索
            df = ak.stock_hk_spot_em()
            results = df[df['代码'].str.contains(keyword) | df['名称'].str.contains(keyword)]
            return results.to_dict('records')
            
            # 美股搜索
            df = ak.stock_us_spot_em()
            results = df[df['代码'].str.contains(keyword) | df['名称'].str.contains(keyword)]
            return results.to_dict('records')
            
        except Exception as e:
            return []
```

#### 2.3 交易日历模块
```python
# trading_calendar.py
import pandas as pd
import datetime
from chinese_calendar import is_workday, is_holiday

class TradingCalendar:
    def __init__(self):
        self.holidays = self.load_holidays()
    
    def is_trading_day(self, date: datetime.date) -> bool:
        """判断是否为交易日"""
        # 检查是否为工作日且非节假日
        return is_workday(date) and not is_holiday(date)
    
    def get_next_trading_day(self, date: datetime.date, n: int = 1) -> datetime.date:
        """获取第n个交易日"""
        current_date = date
        trading_days_found = 0
        
        while trading_days_found < n:
            current_date += datetime.timedelta(days=1)
            if self.is_trading_day(current_date):
                trading_days_found += 1
        
        return current_date
    
    def get_trading_days_range(self, start_date: datetime.date, 
                              end_date: datetime.date) -> list:
        """获取指定日期范围内的交易日列表"""
        trading_days = []
        current_date = start_date
        
        while current_date <= end_date:
            if self.is_trading_day(current_date):
                trading_days.append(current_date)
            current_date += datetime.timedelta(days=1)
        
        return trading_days
    
    def load_holidays(self) -> list:
        """加载节假日数据"""
        # 这里可以集成更多的节假日数据源
        return []
```

### 前端架构改进

#### 3.1 中文界面优化
```javascript
// zh-CN.js - 中文语言包
const zh_CN = {
    // 通用
    common: {
        loading: '加载中...',
        success: '操作成功',
        error: '操作失败',
        confirm: '确认',
        cancel: '取消'
    },
    
    // 导航和标题
    navigation: {
        title: 'Kronos 股票预测系统',
        subtitle: '基于AI的金融K线数据预测分析平台',
        controlPanel: '控制面板',
        predictionChart: '预测结果图表'
    },
    
    // 模型相关
    model: {
        selectModel: '选择模型',
        autoLoad: '自动加载',
        deviceSelect: '选择设备',
        loadModel: '加载模型',
        modelStatus: '模型状态',
        availableModels: '可用模型',
        modelDescription: '模型描述'
    },
    
    // 数据相关
    data: {
        stockCode: '股票代码',
        searchStock: '搜索股票',
        selectData: '选择数据',
        loadData: '加载数据',
        dataInfo: '数据信息',
        timeRange: '时间范围',
        priceRange: '价格范围',
        timeframe: '时间频率'
    },
    
    // 预测相关
    prediction: {
        startPrediction: '开始预测',
        predictionParams: '预测参数',
        lookbackWindow: '回看窗口',
        predictionLength: '预测长度',
        temperature: '温度参数',
        topP: 'Top-P参数',
        sampleCount: '样本数量'
    },
    
    // 图表相关
    chart: {
        title: '股票价格预测',
        xAxis: '时间',
        yAxis: '价格(元)',
        actualData: '实际数据',
        predictedData: '预测数据',
        confidenceInterval: '置信区间'
    }
};

// 语言切换函数
function setLanguage(lang) {
    const translations = lang === 'zh-CN' ? zh_CN : en_US;
    
    // 更新所有界面元素
    document.querySelectorAll('[data-i18n]').forEach(element => {
        const key = element.getAttribute('data-i18n');
        const translation = getNestedValue(translations, key);
        if (translation) {
            element.textContent = translation;
        }
    });
    
    // 更新占位符文本
    document.querySelectorAll('[data-i18n-placeholder]').forEach(element => {
        const key = element.getAttribute('data-i18n-placeholder');
        const translation = getNestedValue(translations, key);
        if (translation) {
            element.placeholder = translation;
        }
    });
}

// 辅助函数：获取嵌套对象值
function getNestedValue(obj, path) {
    return path.split('.').reduce((current, key) => current && current[key], obj);
}
```

#### 3.2 美观化图表配置
```javascript
// chart_config.js
const chartConfig = {
    // 深色主题配置
    darkTheme: {
        layout: {
            title: {
                text: '股票价格预测',
                font: { size: 20, color: '#ffffff' },
                x: 0.5,
                xanchor: 'center'
            },
            plot_bgcolor: '#1e1e1e',
            paper_bgcolor: '#2d2d2d',
            font: { color: '#ffffff' },
            xaxis: {
                title: { text: '时间', font: { size: 14 } },
                gridcolor: '#444444',
                tickformat: '%Y-%m-%d',
                rangeslider: { visible: true }
            },
            yaxis: {
                title: { text: '价格(元)', font: { size: 14 } },
                gridcolor: '#444444',
                tickprefix: '¥'
            },
            legend: {
                orientation: 'h',
                y: -0.2,
                font: { size: 12 }
            },
            margin: { l: 60, r: 40, t: 80, b: 60 }
        },
        
        // 实际数据样式
        actualData: {
            type: 'candlestick',
            name: '实际数据',
            increasing: { line: { color: '#00ff88' } },
            decreasing: { line: { color: '#ff4444' } }
        },
        
        // 预测数据样式
        predictedData: {
            type: 'scatter',
            mode: 'lines+markers',
            name: '预测数据',
            line: { color: '#ffaa00', width: 3 },
            marker: { size: 6, color: '#ffaa00' }
        },
        
        // 置信区间样式
        confidenceInterval: {
            type: 'scatter',
            mode: 'lines',
            name: '置信区间',
            line: { width: 0 },
            fillcolor: 'rgba(255, 170, 0, 0.2)',
            showlegend: false
        }
    },
    
    // 浅色主题配置
    lightTheme: {
        // 类似的浅色主题配置
        layout: {
            title: { text: '股票价格预测', font: { size: 20, color: '#333333' } },
            plot_bgcolor: '#ffffff',
            paper_bgcolor: '#f8f9fa',
            font: { color: '#333333' }
            // ... 其他配置
        }
    }
};

// 图表初始化函数
function initializeChart(containerId, theme = 'dark') {
    const config = chartConfig[theme + 'Theme'];
    
    return Plotly.newPlot(containerId, [], config.layout, {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['pan2d', 'lasso2d'],
        modeBarButtonsToAdd: ['hoverClosestCartesian', 'hoverCompareCartesian']
    });
}
```

## 🔄 API接口设计

### 新增API端点

#### 4.1 股票搜索API
```python
@app.route('/api/search-stock', methods=['GET'])
def search_stock():
    """搜索股票"""
    keyword = request.args.get('keyword', '')
    
    if not keyword:
        return jsonify({'success': False, 'error': '请输入搜索关键词'})
    
    try:
        data_provider = AkshareDataProvider()
        results = data_provider.search_stock(keyword)
        return jsonify({
            'success': True,
            'results': results
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'搜索失败: {str(e)}'
        })
```

#### 4.2 自动模型加载API
```python
@app.route('/api/auto-load-model', methods=['POST'])
def auto_load_model():
    """自动加载最优模型"""
    try:
        auto_loader = AutoModelLoader()
        
        # 选择最优模型和设备
        model_key, model_info = auto_loader.select_optimal_model()
        device = auto_loader.select_optimal_device()
        
        # 加载模型
        global model, tokenizer, predictor
        
        tokenizer = KronosTokenizer.from_pretrained(model_info['tokenizer_id'])
        model = Kronos.from_pretrained(model_info['model_id'])
        predictor = KronosPredictor(model, tokenizer, device=device)
        
        return jsonify({
            'success': True,
            'message': f'模型自动加载成功: {model_info["name"]} on {device}',
            'model_info': {
                'name': model_info['name'],
                'device': device,
                'params': model_info['params']
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'自动模型加载失败: {str(e)}'
        })
```

#### 4.3 交易日历API
```python
@app.route('/api/trading-calendar', methods=['GET'])
def get_trading_calendar():
    """获取交易日历"""
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    try:
        calendar = TradingCalendar()
        
        if start_date and end_date:
            start = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
            end = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
            trading_days = calendar.get_trading_days_range(start, end)
        else:
            # 返回未来30个交易日
            today = datetime.date.today()
            end = today + datetime.timedelta(days=60)  # 足够的天数以包含30个交易日
            trading_days = calendar.get_trading_days_range(today, end)[:30]
        
        return jsonify({
            'success': True,
            'trading_days': [day.isoformat() for day in trading_days]
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'获取交易日历失败: {str(e)}'
        })
```

## 📊 数据库设计

### 5.1 数据缓存表
```sql
-- 股票数据缓存表
CREATE TABLE stock_data_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol VARCHAR(20) NOT NULL,
    period VARCHAR(10) NOT NULL,
    start_date DATE,
    end_date DATE,
    data_json TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, period, start_date, end_date)
);

-- 预测结果表
CREATE TABLE prediction_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol VARCHAR(20) NOT NULL,
    prediction_type VARCHAR(50) NOT NULL,
    prediction_data JSON NOT NULL,
    parameters JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 用户配置表
CREATE TABLE user_settings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    language VARCHAR(10) DEFAULT 'zh-CN',
    theme VARCHAR(10) DEFAULT 'dark',
    auto_model_loading BOOLEAN DEFAULT TRUE,
    default_device VARCHAR(10) DEFAULT 'auto',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 🚀 部署和配置

### 6.1 依赖包更新
```txt
# requirements.txt 新增依赖
akshare>=1.10.0
chinese-calendar>=1.8.0
plotly>=5.17.0
psutil>=5.9.0
```

### 6.2 配置文件
```python
# config.py
class Config:
    # 应用配置
    DEBUG = False
    SECRET_KEY = 'your-secret-key-here'
    
    # 模型配置
    AUTO_MODEL_LOADING = True
    DEFAULT_DEVICE = 'auto'
    MODEL_CACHE_DIR = './model_cache'
    
    # 数据源配置
    DATA_SOURCE = 'akshare'  # 'local' or 'akshare'
    CACHE_DURATION = 3600  # 数据缓存时间(秒)
    MAX_CACHE_SIZE = 1000  # 最大缓存条目数
    
    # 界面配置
    DEFAULT_LANGUAGE = 'zh-CN'  # 'en-US' or 'zh-CN'
    DEFAULT_THEME = 'dark'  # 'light' or 'dark'
    
    # 预测配置
    DEFAULT_LOOKBACK = 400
    DEFAULT_PREDICTION_LENGTH = 120
    MAX_PREDICTION_LENGTH = 365
    
    # 性能配置
    MAX_CONCURRENT_REQUESTS = 10
    REQUEST_TIMEOUT = 30
```

## 📈 性能优化策略

### 7.1 前端性能优化
- 使用CDN加载静态资源
- 实现懒加载和代码分割
- 优化图片和图表资源
- 使用Service Worker缓存

### 7.2 后端性能优化
- 实现数据缓存机制
- 使用异步任务处理
- 数据库查询优化
- 内存使用监控

### 7.3 预测性能优化
- 模型预加载和缓存
- 批量预测处理
- GPU加速优化
- 预测结果缓存

## 🔍 测试策略

### 8.1 单元测试
```python
# test_auto_model_loader.py
import unittest
from auto_model_loader import AutoModelLoader

class TestAutoModelLoader(unittest.TestCase):
    def setUp(self):
        self.loader = AutoModelLoader()
    
    def test_detect_system_resources(self):
        resources = self.loader.detect_system_resources()
        self.assertIn('memory_gb', resources)
        self.assertIn('gpu_available', resources)
    
    def test_select_optimal_model(self):
        model_key, model_info = self.loader.select_optimal_model()
        self.assertIn(model_key, self.loader.available_models)
        self.assertIsNotNone(model_info)
```

### 8.2 集成测试
- API接口测试
- 端到端业务流程测试
- 性能压力测试
- 兼容性测试

## 📚 文档计划

### 9.1 用户文档
- 快速入门指南
- 功能使用说明
- 常见问题解答

### 9.2 开发文档
- API接口文档
- 代码架构说明
- 部署指南

### 9.3 运维文档
- 系统监控指南
- 故障排查手册
- 性能优化建议

## 🎯 验收标准

### 功能验收
- [ ] 自动模型加载功能正常
- [ ] akshare数据获取准确
- [ ] 中文界面显示正确
- [ ] 预测图表美观可用
- [ ] 交易日历功能正常

### 性能验收
- [ ] 系统响应时间 < 3秒
- [ ] 模型加载时间 < 10秒
- [ ] 数据获取时间 < 2秒
- [ ] 预测时间 < 5秒

### 用户体验验收
- [ ] 操作流程简洁明了
- [ ] 界面布局合理美观
- [ ] 错误提示友好清晰
- [ ] 响应式设计完善

---

*本文档将根据开发进度实时更新，最新版本请参考GitHub仓库*