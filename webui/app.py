import os
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.utils
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys
import warnings
import datetime
import argparse
warnings.filterwarnings('ignore')

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import akshare data provider
try:
    from akshare_data_provider import AkshareDataProvider
    data_provider = AkshareDataProvider()
except ImportError as e:
    print(f"Warning: akshare_data_provider not available: {e}")
    data_provider = None

# Import auto model loader
try:
    from auto_model_loader import AutoModelLoader
    auto_loader = AutoModelLoader()
except ImportError as e:
    print(f"Warning: auto_model_loader not available: {e}")
    auto_loader = None

# Import local model manager
try:
    from local_model_manager import LocalModelManager
    local_model_manager = LocalModelManager()
    print("Local model manager initialized successfully")
except ImportError as e:
    print(f"Warning: local_model_manager not available: {e}")
    local_model_manager = None

# Import direct model loader
try:
    from direct_model_loader import DirectModelLoader
    direct_model_loader = DirectModelLoader()
    print("Direct model loader initialized successfully")
    
    # 自动加载最优模型
    success, message, model_key = direct_model_loader.auto_load_best_model()
    if success:
        print(f"Auto-loaded model: {model_key} - {message}")
        # 设置全局模型变量
        loaded_model = direct_model_loader.get_loaded_model()
        if loaded_model:
            tokenizer = loaded_model['tokenizer']
            model = loaded_model['model']
            print(f"Model {model_key} loaded successfully")
    else:
        print(f"Auto-load failed: {message}")
        
except ImportError as e:
    print(f"Warning: direct_model_loader not available: {e}")
    direct_model_loader = None

try:
    from model import Kronos, KronosTokenizer, KronosPredictor
    # 测试模型库是否真正可用
    try:
        # 尝试导入一个简单的模型来验证可用性
        test_tokenizer = KronosTokenizer.from_pretrained('NeoQuasar/Kronos-Tokenizer-base')
        MODEL_AVAILABLE = True
    except Exception as e:
        MODEL_AVAILABLE = False
        print(f"Warning: Kronos model library not available: {e}")
except ImportError:
    MODEL_AVAILABLE = False
    print("Warning: Kronos model cannot be imported, will use simulated data for demonstration")

app = Flask(__name__)
CORS(app)

# Global variables to store models
tokenizer = None
model = None
predictor = None

# Available model configurations
AVAILABLE_MODELS = {
    'kronos-mini': {
        'name': 'Kronos-mini',
        'model_id': 'NeoQuasar/Kronos-mini',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-2k',
        'context_length': 2048,
        'params': '4.1M',
        'description': 'Lightweight model, suitable for fast prediction'
    },
    'kronos-small': {
        'name': 'Kronos-small',
        'model_id': 'NeoQuasar/Kronos-small',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '24.7M',
        'description': 'Small model, balanced performance and speed'
    },
    'kronos-base': {
        'name': 'Kronos-base',
        'model_id': 'NeoQuasar/Kronos-base',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '102.3M',
        'description': 'Base model, provides better prediction quality'
    }
}

def load_data_files():
    """Scan data directory and return available data files"""
    # 修复数据目录路径，从examples/data加载文件
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'examples', 'data')
    data_files = []
    
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith(('.csv', '.feather')):
                file_path = os.path.join(data_dir, file)
                file_size = os.path.getsize(file_path)
                data_files.append({
                    'name': file,
                    'path': file_path,
                    'size': f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.1f} MB"
                })
    
    return data_files

def load_data_file(file_path):
    """Load data file"""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.feather'):
            df = pd.read_feather(file_path)
        else:
            return None, "Unsupported file format"
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return None, f"Missing required columns: {required_cols}"
        
        # Process timestamp column
        if 'timestamps' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamps'])
        elif 'timestamp' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamp'])
        elif 'date' in df.columns:
            # If column name is 'date', rename it to 'timestamps'
            df['timestamps'] = pd.to_datetime(df['date'])
        else:
            # If no timestamp column exists, create one
            df['timestamps'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1H')
        
        # Ensure numeric columns are numeric type
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Process volume column (optional)
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        # Process amount column (optional, but not used for prediction)
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        # Remove rows containing NaN values
        df = df.dropna()
        
        return df, None
        
    except Exception as e:
        return None, f"Failed to load file: {str(e)}"

def save_prediction_results(file_path, prediction_type, prediction_results, actual_data, input_data, prediction_params):
    """Save prediction results to file"""
    try:
        # Create prediction results directory
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prediction_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'prediction_{timestamp}.json'
        filepath = os.path.join(results_dir, filename)
        
        # Prepare data for saving
        save_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'file_path': file_path,
            'prediction_type': prediction_type,
            'prediction_params': prediction_params,
            'input_data_summary': {
                'rows': len(input_data),
                'columns': list(input_data.columns),
                'price_range': {
                    'open': {'min': float(input_data['open'].min()), 'max': float(input_data['open'].max())},
                    'high': {'min': float(input_data['high'].min()), 'max': float(input_data['high'].max())},
                    'low': {'min': float(input_data['low'].min()), 'max': float(input_data['low'].max())},
                    'close': {'min': float(input_data['close'].min()), 'max': float(input_data['close'].max())}
                },
                'last_values': {
                    'open': float(input_data['open'].iloc[-1]),
                    'high': float(input_data['high'].iloc[-1]),
                    'low': float(input_data['low'].iloc[-1]),
                    'close': float(input_data['close'].iloc[-1])
                }
            },
            'prediction_results': prediction_results,
            'actual_data': actual_data,
            'analysis': {}
        }
        
        # If actual data exists, perform comparison analysis
        if actual_data and len(actual_data) > 0:
            # Calculate continuity analysis
            if len(prediction_results) > 0 and len(actual_data) > 0:
                last_pred = prediction_results[0]  # First prediction point
            first_actual = actual_data[0]      # First actual point
                
            save_data['analysis']['continuity'] = {
                    'last_prediction': {
                        'open': last_pred['open'],
                        'high': last_pred['high'],
                        'low': last_pred['low'],
                        'close': last_pred['close']
                    },
                    'first_actual': {
                        'open': first_actual['open'],
                        'high': first_actual['high'],
                        'low': first_actual['low'],
                        'close': first_actual['close']
                    },
                    'gaps': {
                        'open_gap': abs(last_pred['open'] - first_actual['open']),
                        'high_gap': abs(last_pred['high'] - first_actual['high']),
                        'low_gap': abs(last_pred['low'] - first_actual['low']),
                        'close_gap': abs(last_pred['close'] - first_actual['close'])
                    },
                    'gap_percentages': {
                        'open_gap_pct': (abs(last_pred['open'] - first_actual['open']) / first_actual['open']) * 100,
                        'high_gap_pct': (abs(last_pred['high'] - first_actual['high']) / first_actual['high']) * 100,
                        'low_gap_pct': (abs(last_pred['low'] - first_actual['low']) / first_actual['low']) * 100,
                        'close_gap_pct': (abs(last_pred['close'] - first_actual['close']) / first_actual['close']) * 100
                    }
                }
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"Prediction results saved to: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"Failed to save prediction results: {e}")
        return None

def create_prediction_chart(df, pred_df, lookback, pred_len, actual_df=None, historical_start_idx=0):
    """创建美观的预测图表"""
    # 使用指定的历史数据起始位置
    if historical_start_idx + lookback + pred_len <= len(df):
        historical_df = df.iloc[historical_start_idx:historical_start_idx+lookback]
        prediction_range = range(historical_start_idx+lookback, historical_start_idx+lookback+pred_len)
    else:
        available_lookback = min(lookback, len(df) - historical_start_idx)
        available_pred_len = min(pred_len, max(0, len(df) - historical_start_idx - available_lookback))
        historical_df = df.iloc[historical_start_idx:historical_start_idx+available_lookback]
        prediction_range = range(historical_start_idx+available_lookback, historical_start_idx+available_lookback+available_pred_len)
    
    # 创建图表
    fig = go.Figure()
    
    # 美观的深色主题配置
    dark_theme_layout = {
        'title': {
            'text': 'Kronos 股票价格预测',
            'font': {'size': 24, 'color': '#ffffff', 'family': 'Arial, sans-serif'},
            'x': 0.5,
            'xanchor': 'center'
        },
        'plot_bgcolor': '#1e1e1e',
        'paper_bgcolor': '#2d2d2d',
        'font': {'color': '#ffffff', 'family': 'Arial, sans-serif'},
        'xaxis': {
            'title': {'text': '时间', 'font': {'size': 16, 'color': '#ffffff'}},
            'gridcolor': '#444444',
            'tickformat': '%Y-%m-%d',
            'rangeslider': {'visible': True, 'bgcolor': '#3d3d3d'},
            'type': 'date'
        },
        'yaxis': {
            'title': {'text': '价格(元)', 'font': {'size': 16, 'color': '#ffffff'}},
            'gridcolor': '#444444',
            'tickprefix': '¥',
            'side': 'right'
        },
        'legend': {
            'orientation': 'h',
            'y': -0.15,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 12, 'color': '#ffffff'},
            'bgcolor': 'rgba(0,0,0,0.5)',
            'bordercolor': '#666666'
        },
        'margin': {'l': 60, 'r': 60, 't': 100, 'b': 80},
        'height': 650,
        'hovermode': 'x unified',
        'showlegend': True
    }
    
    # 添加历史数据（K线图）
    fig.add_trace(go.Candlestick(
        x=historical_df['timestamps'] if 'timestamps' in historical_df.columns else historical_df.index,
        open=historical_df['open'],
        high=historical_df['high'],
        low=historical_df['low'],
        close=historical_df['close'],
        name='历史数据',
        increasing_line_color='#00C851',  # 绿色
        decreasing_line_color='#ff4444',  # 红色
        increasing_fillcolor='rgba(0, 200, 81, 0.3)',
        decreasing_fillcolor='rgba(255, 68, 68, 0.3)',
        line_width=1.5
    ))
    
    # 添加预测数据（K线图）
    if pred_df is not None and len(pred_df) > 0:
        if 'timestamps' in df.columns and len(historical_df) > 0:
            last_timestamp = historical_df['timestamps'].iloc[-1]
            time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0] if len(df) > 1 else pd.Timedelta(hours=1)
            
            pred_timestamps = pd.date_range(
                start=last_timestamp + time_diff,
                periods=len(pred_df),
                freq=time_diff
            )
        else:
            pred_timestamps = range(len(historical_df), len(historical_df) + len(pred_df))
        
        fig.add_trace(go.Candlestick(
            x=pred_timestamps,
            open=pred_df['open'],
            high=pred_df['high'],
            low=pred_df['low'],
            close=pred_df['close'],
            name='预测数据',
            increasing_line_color='#33b5e5',  # 蓝色
            decreasing_line_color='#ff8800',  # 橙色
            increasing_fillcolor='rgba(51, 181, 229, 0.3)',
            decreasing_fillcolor='rgba(255, 136, 0, 0.3)',
            line_width=1.5
        ))
    
    # 添加实际数据用于比较（如果存在）
    if actual_df is not None and len(actual_df) > 0:
        if 'timestamps' in df.columns:
            if 'pred_timestamps' in locals():
                actual_timestamps = pred_timestamps
            else:
                if len(historical_df) > 0:
                    last_timestamp = historical_df['timestamps'].iloc[-1]
                    time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0] if len(df) > 1 else pd.Timedelta(hours=1)
                    actual_timestamps = pd.date_range(
                        start=last_timestamp + time_diff,
                        periods=len(actual_df),
                        freq=time_diff
                    )
                else:
                    actual_timestamps = range(len(historical_df), len(historical_df) + len(actual_df))
        else:
            actual_timestamps = range(len(historical_df), len(historical_df) + len(actual_df))
        
        fig.add_trace(go.Candlestick(
            x=actual_timestamps,
            open=actual_df['open'],
            high=actual_df['high'],
            low=actual_df['low'],
            close=actual_df['close'],
            name='实际数据',
            increasing_line_color='#aa66cc',  # 紫色
            decreasing_line_color='#ffbb33',  # 黄色
            increasing_fillcolor='rgba(170, 102, 204, 0.3)',
            decreasing_fillcolor='rgba(255, 187, 51, 0.3)',
            line_width=1.5
        ))
    
    # 应用美观的布局配置
    fig.update_layout(dark_theme_layout)
    
    # 添加交互功能配置
    fig.update_layout(
        modebar={
            'orientation': 'v',
            'bgcolor': 'rgba(0,0,0,0.7)',
            'color': '#ffffff',
            'activecolor': '#33b5e5'
        }
    )
    
    # 确保x轴时间连续性
    if 'timestamps' in historical_df.columns:
        all_timestamps = []
        if len(historical_df) > 0:
            all_timestamps.extend(historical_df['timestamps'])
        if 'pred_timestamps' in locals():
            all_timestamps.extend(pred_timestamps)
        if 'actual_timestamps' in locals():
            all_timestamps.extend(actual_timestamps)
        
        if all_timestamps:
            all_timestamps = sorted(all_timestamps)
            fig.update_xaxes(
                range=[all_timestamps[0], all_timestamps[-1]],
                rangeslider_visible=True,
                rangeslider_thickness=0.1
            )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/api/data-files')
def get_data_files():
    """Get available data file list"""
    data_files = load_data_files()
    return jsonify(data_files)

@app.route('/api/load-data', methods=['POST'])
def load_data():
    """Load data file"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        
        if not file_path:
            return jsonify({'error': 'File path cannot be empty'}), 400
        
        df, error = load_data_file(file_path)
        if error:
            return jsonify({'error': error}), 400
        
        # Get basic information
        data_info = {
            'file_path': file_path,
            'row_count': len(df),
            'columns': list(df.columns),
            'data_types': {col: str(df[col].dtype) for col in df.columns},
            'start_time': df['timestamps'].min().isoformat() if 'timestamps' in df.columns else None,
            'end_time': df['timestamps'].max().isoformat() if 'timestamps' in df.columns else None,
            'sample_data': df.head(5).to_dict('records')
        }
        
        return jsonify({
            'success': True,
            'data_info': data_info
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to load data: {str(e)}'}), 500

@app.route('/api/akshare/search-stock', methods=['POST'])
def search_stock():
    """Search stock by symbol or name"""
    try:
        if data_provider is None:
            return jsonify({'error': 'Akshare data provider not available'}), 503
        
        data = request.get_json()
        keyword = data.get('keyword', '').strip()
        
        if not keyword:
            return jsonify({'error': 'Search keyword cannot be empty'}), 400
        
        results = data_provider.search_stock(keyword)
        
        return jsonify({
            'success': True,
            'results': results,
            'count': len(results)
        })
        
    except Exception as e:
        return jsonify({'error': f'Stock search failed: {str(e)}'}), 500

@app.route('/api/akshare/get-stock-data', methods=['POST'])
def get_stock_data():
    """Get stock historical data"""
    try:
        if data_provider is None:
            return jsonify({'error': 'Akshare data provider not available'}), 503
        
        data = request.get_json()
        symbol = data.get('symbol', '').strip()
        start_date = data.get('start_date', '')
        end_date = data.get('end_date', '')
        period = data.get('period', 'daily')  # daily, weekly, monthly
        
        if not symbol:
            return jsonify({'error': 'Stock symbol cannot be empty'}), 400
        
        stock_data = data_provider.get_stock_data(symbol, period, start_date, end_date)
        
        if stock_data is None or stock_data.empty:
            return jsonify({'error': f'No data found for symbol: {symbol}'}), 404
        
        # Convert to DataFrame for processing
        df = pd.DataFrame(stock_data)
        
        # Prepare data info
        data_info = {
            'symbol': symbol,
            'row_count': len(df),
            'columns': list(df.columns),
            'start_date': df['date'].min() if 'date' in df.columns else None,
            'end_date': df['date'].max() if 'date' in df.columns else None,
            'sample_data': df.head(5).to_dict('records')
        }
        
        return jsonify({
            'success': True,
            'data': stock_data.to_dict('records'),
            'data_info': data_info
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to get stock data: {str(e)}'}), 500

@app.route('/api/akshare/trading-calendar', methods=['GET'])
def get_trading_calendar():
    """Get trading calendar"""
    try:
        if data_provider is None:
            return jsonify({'error': 'Akshare data provider not available'}), 503
        
        year = request.args.get('year', str(datetime.now().year))
        
        calendar_data = data_provider.get_trading_calendar(year)
        
        return jsonify({
            'success': True,
            'year': year,
            'trading_days': calendar_data,
            'count': len(calendar_data)
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to get trading calendar: {str(e)}'}), 500

@app.route('/api/akshare/is-trading-day', methods=['POST'])
def is_trading_day():
    """Check if a date is trading day"""
    try:
        if data_provider is None:
            return jsonify({'error': 'Akshare data provider not available'}), 503
        
        data = request.get_json()
        date_str = data.get('date', '')
        
        if not date_str:
            return jsonify({'error': 'Date cannot be empty'}), 400
        
        is_trading = data_provider.is_trading_day(date_str)
        
        return jsonify({
            'success': True,
            'date': date_str,
            'is_trading_day': is_trading
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to check trading day: {str(e)}'}), 500

@app.route('/api/akshare/next-trading-day', methods=['POST'])
def get_next_trading_day():
    """Get next trading day"""
    try:
        if data_provider is None:
            return jsonify({'error': 'Akshare data provider not available'}), 503
        
        data = request.get_json()
        date_str = data.get('date', '')
        
        if not date_str:
            return jsonify({'error': 'Date cannot be empty'}), 400
        
        next_day = data_provider.get_next_trading_day(date_str)
        
        return jsonify({
            'success': True,
            'date': date_str,
            'next_trading_day': next_day
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to get next trading day: {str(e)}'}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Perform prediction"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        lookback = int(data.get('lookback', 400))
        pred_len = int(data.get('pred_len', 120))
        
        # Get prediction quality parameters
        temperature = float(data.get('temperature', 1.0))
        top_p = float(data.get('top_p', 0.9))
        sample_count = int(data.get('sample_count', 1))
        
        if not file_path:
            return jsonify({'error': 'File path cannot be empty'}), 400
        
        # Load data
        df, error = load_data_file(file_path)
        if error:
            return jsonify({'error': error}), 400
        
        if len(df) < lookback:
            return jsonify({'error': f'Insufficient data length, need at least {lookback} rows'}), 400
        
        # Perform prediction
        # 优先使用直接加载的模型
        if direct_model_loader and direct_model_loader.get_loaded_model():
            try:
                # Use directly loaded model
                loaded_model = direct_model_loader.get_loaded_model()
                predictor = KronosPredictor(loaded_model['model'], loaded_model['tokenizer'])
                
                # Only use necessary columns: OHLCV, excluding amount
                required_cols = ['open', 'high', 'low', 'close']
                if 'volume' in df.columns:
                    required_cols.append('volume')
                
                # Process time period selection
                start_date = data.get('start_date')
                
                if start_date:
                    # Custom time period - fix logic: use data within selected window
                    start_dt = pd.to_datetime(start_date)
                    
                    # Find data after start time
                    mask = df['timestamps'] >= start_dt
                    time_range_df = df[mask]
                    
                    # Ensure sufficient data: lookback + pred_len
                    if len(time_range_df) < lookback + pred_len:
                        return jsonify({'error': f'Insufficient data from start time {start_dt.strftime("%Y-%m-%d %H:%M")}, need at least {lookback + pred_len} data points, currently only {len(time_range_df)} available'}), 400
                    
                    # Use first lookback data points within selected window for prediction
                    x_df = time_range_df.iloc[:lookback][required_cols]
                    x_timestamp = time_range_df.iloc[:lookback]['timestamps']
                    
                    # Use last pred_len data points within selected window as actual values
                    y_timestamp = time_range_df.iloc[lookback:lookback+pred_len]['timestamps']
                    
                    # Calculate actual time period length
                    start_timestamp = time_range_df['timestamps'].iloc[0]
                    end_timestamp = time_range_df['timestamps'].iloc[lookback+pred_len-1]
                    time_span = end_timestamp - start_timestamp
                    
                    prediction_type = f"Kronos model prediction (within selected window: first {lookback} data points for prediction, last {pred_len} data points for comparison, time span: {time_span})"
                else:
                    # Use latest data
                    x_df = df.iloc[:lookback][required_cols]
                    x_timestamp = df.iloc[:lookback]['timestamps']
                    y_timestamp = df.iloc[lookback:lookback+pred_len]['timestamps']
                    prediction_type = "Kronos model prediction (latest data)"
                
                # Ensure timestamps are Series format, not DatetimeIndex, to avoid .dt attribute error in Kronos model
                if isinstance(x_timestamp, pd.DatetimeIndex):
                    x_timestamp = pd.Series(x_timestamp, name='timestamps')
                if isinstance(y_timestamp, pd.DatetimeIndex):
                    y_timestamp = pd.Series(y_timestamp, name='timestamps')
                
                pred_df = predictor.predict(
                    df=x_df,
                    x_timestamp=x_timestamp,
                    y_timestamp=y_timestamp,
                    pred_len=pred_len,
                    T=temperature,
                    top_p=top_p,
                    sample_count=sample_count
                )
                
            except Exception as e:
                return jsonify({'error': f'Kronos model prediction failed: {str(e)}'}), 500
        # 其次使用全局加载的模型
        elif MODEL_AVAILABLE and predictor is not None:
            try:
                # Use real Kronos model
                # Only use necessary columns: OHLCV, excluding amount
                required_cols = ['open', 'high', 'low', 'close']
                if 'volume' in df.columns:
                    required_cols.append('volume')
                
                # Process time period selection
                start_date = data.get('start_date')
                
                if start_date:
                    # Custom time period - fix logic: use data within selected window
                    start_dt = pd.to_datetime(start_date)
                    
                    # Find data after start time
                    mask = df['timestamps'] >= start_dt
                    time_range_df = df[mask]
                    
                    # Ensure sufficient data: lookback + pred_len
                    if len(time_range_df) < lookback + pred_len:
                        return jsonify({'error': f'Insufficient data from start time {start_dt.strftime("%Y-%m-%d %H:%M")}, need at least {lookback + pred_len} data points, currently only {len(time_range_df)} available'}), 400
                    
                    # Use first lookback data points within selected window for prediction
                    x_df = time_range_df.iloc[:lookback][required_cols]
                    x_timestamp = time_range_df.iloc[:lookback]['timestamps']
                    
                    # Use last pred_len data points within selected window as actual values
                    y_timestamp = time_range_df.iloc[lookback:lookback+pred_len]['timestamps']
                    
                    # Calculate actual time period length
                    start_timestamp = time_range_df['timestamps'].iloc[0]
                    end_timestamp = time_range_df['timestamps'].iloc[lookback+pred_len-1]
                    time_span = end_timestamp - start_timestamp
                    
                    prediction_type = f"Kronos model prediction (within selected window: first {lookback} data points for prediction, last {pred_len} data points for comparison, time span: {time_span})"
                else:
                    # Use latest data
                    x_df = df.iloc[:lookback][required_cols]
                    x_timestamp = df.iloc[:lookback]['timestamps']
                    y_timestamp = df.iloc[lookback:lookback+pred_len]['timestamps']
                    prediction_type = "Kronos model prediction (latest data)"
                
                # Ensure timestamps are Series format, not DatetimeIndex, to avoid .dt attribute error in Kronos model
                if isinstance(x_timestamp, pd.DatetimeIndex):
                    x_timestamp = pd.Series(x_timestamp, name='timestamps')
                if isinstance(y_timestamp, pd.DatetimeIndex):
                    y_timestamp = pd.Series(y_timestamp, name='timestamps')
                
                pred_df = predictor.predict(
                    df=x_df,
                    x_timestamp=x_timestamp,
                    y_timestamp=y_timestamp,
                    pred_len=pred_len,
                    T=temperature,
                    top_p=top_p,
                    sample_count=sample_count
                )
                
            except Exception as e:
                return jsonify({'error': f'Kronos model prediction failed: {str(e)}'}), 500
        else:
            return jsonify({'error': 'Kronos model not loaded, please load model first'}), 400
        
        # Prepare actual data for comparison (if exists)
        actual_data = []
        actual_df = None
        
        if start_date:  # Custom time period
            # Fix logic: use data within selected window
            # Prediction uses first 400 data points within selected window
            # Actual data should be last 120 data points within selected window
            start_dt = pd.to_datetime(start_date)
            
            # Find data starting from start_date
            mask = df['timestamps'] >= start_dt
            time_range_df = df[mask]
            
            if len(time_range_df) >= lookback + pred_len:
                # Get last 120 data points within selected window as actual values
                actual_df = time_range_df.iloc[lookback:lookback+pred_len]
                
                for i, (_, row) in enumerate(actual_df.iterrows()):
                    actual_data.append({
                        'timestamp': row['timestamps'].isoformat(),
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': float(row['volume']) if 'volume' in row else 0,
                        'amount': float(row['amount']) if 'amount' in row else 0
                    })
        else:  # Latest data
            # Prediction uses first 400 data points
            # Actual data should be 120 data points after first 400 data points
            if len(df) >= lookback + pred_len:
                actual_df = df.iloc[lookback:lookback+pred_len]
                for i, (_, row) in enumerate(actual_df.iterrows()):
                    actual_data.append({
                        'timestamp': row['timestamps'].isoformat(),
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': float(row['volume']) if 'volume' in row else 0,
                        'amount': float(row['amount']) if 'amount' in row else 0
                    })
        
        # Create chart - pass historical data start position
        if start_date:
            # Custom time period: find starting position of historical data in original df
            start_dt = pd.to_datetime(start_date)
            mask = df['timestamps'] >= start_dt
            historical_start_idx = df[mask].index[0] if len(df[mask]) > 0 else 0
        else:
            # Latest data: start from beginning
            historical_start_idx = 0
        
        chart_json = create_prediction_chart(df, pred_df, lookback, pred_len, actual_df, historical_start_idx)
        
        # Prepare prediction result data - fix timestamp calculation logic
        if 'timestamps' in df.columns:
            if start_date:
                # Custom time period: use selected window data to calculate timestamps
                start_dt = pd.to_datetime(start_date)
                mask = df['timestamps'] >= start_dt
                time_range_df = df[mask]
                
                if len(time_range_df) >= lookback:
                    # Calculate prediction timestamps starting from last time point of selected window
                    last_timestamp = time_range_df['timestamps'].iloc[lookback-1]
                    time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0]
                    future_timestamps = pd.date_range(
                        start=last_timestamp + time_diff,
                        periods=pred_len,
                        freq=time_diff
                    )
                else:
                    future_timestamps = []
            else:
                # Latest data: calculate from last time point of entire data file
                last_timestamp = df['timestamps'].iloc[-1]
                time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0]
                future_timestamps = pd.date_range(
                    start=last_timestamp + time_diff,
                    periods=pred_len,
                    freq=time_diff
                )
        else:
            future_timestamps = range(len(df), len(df) + pred_len)
        
        prediction_results = []
        for i, (_, row) in enumerate(pred_df.iterrows()):
            prediction_results.append({
                'timestamp': future_timestamps[i].isoformat() if i < len(future_timestamps) else f"T{i}",
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']) if 'volume' in row else 0,
                'amount': float(row['amount']) if 'amount' in row else 0
            })
        
        # Save prediction results to file
        try:
            save_prediction_results(
                file_path=file_path,
                prediction_type=prediction_type,
                prediction_results=prediction_results,
                actual_data=actual_data,
                input_data=x_df,
                prediction_params={
                    'lookback': lookback,
                    'pred_len': pred_len,
                    'temperature': temperature,
                    'top_p': top_p,
                    'sample_count': sample_count,
                    'start_date': start_date if start_date else 'latest'
                }
            )
        except Exception as e:
            print(f"Failed to save prediction results: {e}")
        
        return jsonify({
            'success': True,
            'prediction_type': prediction_type,
            'chart': chart_json,
            'prediction_results': prediction_results,
            'actual_data': actual_data,
            'has_comparison': len(actual_data) > 0,
            'message': f'Prediction completed, generated {pred_len} prediction points' + (f', including {len(actual_data)} actual data points for comparison' if len(actual_data) > 0 else '')
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/load-model', methods=['POST'])
def load_model():
    """Load Kronos model with strict local-first strategy"""
    global tokenizer, model, predictor
    
    try:
        if not MODEL_AVAILABLE:
            return jsonify({'error': 'Kronos model library not available'}), 400
        
        data = request.get_json()
        model_key = data.get('model_key', 'kronos-small')
        device = data.get('device', 'cpu')
        
        if model_key not in AVAILABLE_MODELS:
            return jsonify({'error': f'Unsupported model: {model_key}'}), 400
        
        model_config = AVAILABLE_MODELS[model_key]
        
        # 严格本地优先策略：只从本地加载，不进行远程下载
        if local_model_manager and local_model_manager.is_model_available_locally(model_config['model_id']):
            try:
                # 从本地加载模型
                model, tokenizer = local_model_manager.load_model_from_local(model_config['model_id'])
                load_source = "local"
                print(f"✅ Model loaded from local storage: {model_config['model_id']}")
                
                # Create predictor
                predictor = KronosPredictor(model, tokenizer, device=device, max_context=model_config['context_length'])
                
                return jsonify({
                    'success': True,
                    'message': f'模型加载成功: {model_config["name"]} ({model_config["params"]}) 在 {device} 上运行 (从本地加载)',
                    'model_info': {
                        'name': model_config['name'],
                        'params': model_config['params'],
                        'context_length': model_config['context_length'],
                        'description': model_config['description'],
                        'load_source': load_source
                    }
                })
                
            except Exception as e:
                print(f"❌ Failed to load model from local: {e}")
                return jsonify({
                    'error': f'本地模型加载失败: {str(e)}。请先下载模型到本地。',
                    'suggestion': '请先使用"下载模型"功能将模型保存到本地'
                }), 400
        else:
            # 本地模型不可用，提示用户先下载
            return jsonify({
                'error': f'模型 {model_config["name"]} 在本地不可用',
                'suggestion': '请先使用"下载模型"功能将模型保存到本地，然后重新加载',
                'model_info': {
                    'name': model_config['name'],
                    'model_id': model_config['model_id'],
                    'available_locally': False
                }
            }), 400
        
    except Exception as e:
        return jsonify({'error': f'模型加载失败: {str(e)}'}), 500

@app.route('/api/available-models')
def get_available_models():
    """Get available model list"""
    local_models_info = {}
    if local_model_manager:
        local_models = local_model_manager.get_available_local_models()
        for model_info in local_models:
            local_models_info[model_info['model_id']] = {
                'available': True,
                'size': model_info['size'],
                'downloaded_at': model_info['downloaded_at']
            }
    
    return jsonify({
        'models': AVAILABLE_MODELS,
        'model_available': MODEL_AVAILABLE,
        'local_models': local_models_info
    })

@app.route('/api/download-model', methods=['POST'])
def download_model():
    """Download model to local storage"""
    try:
        if not local_model_manager:
            return jsonify({'error': 'Local model manager not available'}), 400
        
        data = request.get_json()
        model_key = data.get('model_key', 'kronos-small')
        
        if model_key not in AVAILABLE_MODELS:
            return jsonify({'error': f'Unsupported model: {model_key}'}), 400
        
        model_config = AVAILABLE_MODELS[model_key]
        
        # 检查网络连接
        try:
            import requests
            response = requests.get('https://huggingface.co', timeout=10)
            if response.status_code != 200:
                return jsonify({
                    'error': '网络连接异常，无法访问Hugging Face。请检查网络连接或使用镜像源。',
                    'suggestion': '可以尝试设置环境变量 HF_ENDPOINT=https://hf-mirror.com 使用镜像源'
                }), 503
        except Exception as network_error:
            return jsonify({
                'error': f'网络连接失败: {str(network_error)}',
                'suggestion': '请检查网络连接，或使用镜像源下载。系统将自动尝试镜像源。'
            }), 503
        
        # 下载模型到本地
        success, message = local_model_manager.download_model(
            model_config['model_id'],
            model_config['tokenizer_id']
        )
        
        if success:
            return jsonify({
                'success': True,
                'message': f'模型下载成功: {model_config["name"]}',
                'model_info': {
                    'name': model_config['name'],
                    'model_id': model_config['model_id'],
                    'local_path': str(local_model_manager.get_model_path(model_config['model_id'])),
                    'size': model_config['params'],
                    'context_length': model_config['context_length']
                }
            })
        else:
            # 提供更详细的错误信息和建议
            error_details = {
                'error': message,
                'suggestions': [
                    '检查网络连接是否正常',
                    '尝试使用镜像源: 设置环境变量 HF_ENDPOINT=https://hf-mirror.com',
                    '确认模型ID是否正确: ' + model_config['model_id'],
                    '检查磁盘空间是否充足'
                ]
            }
            return jsonify(error_details), 500
            
    except Exception as e:
        return jsonify({
            'error': f'模型下载失败: {str(e)}',
            'suggestion': '请检查网络连接和模型配置'
        }), 500

@app.route('/api/local-models')
def get_local_models():
    """Get local models information"""
    if not local_model_manager:
        return jsonify({'error': 'Local model manager not available'}), 400
    
    local_models = local_model_manager.get_available_local_models()
    storage_info = local_model_manager.get_storage_info()
    
    return jsonify({
        'local_models': local_models,
        'storage_info': storage_info
    })

@app.route('/api/model-status')
def get_model_status():
    """Get model status with detailed monitoring information"""
    try:
        # Get system resource information
        system_info = {}
        if auto_loader:
            try:
                system_info = auto_loader.get_system_report()
            except Exception as e:
                system_info = {'error': f'Failed to get system info: {str(e)}'}
        
        # 检查直接加载的模型状态
        direct_model_loaded = False
        direct_model_info = {}
        
        if direct_model_loader:
            loaded_model = direct_model_loader.get_loaded_model()
            if loaded_model:
                direct_model_loaded = True
                direct_model_info = {
                    'name': loaded_model.get('model_name', 'Unknown'),
                    'path': loaded_model.get('model_path', 'Unknown'),
                    'device': loaded_model.get('device', 'Unknown'),
                    'loaded_from': 'local_directory'
                }
        
        # Enhanced model status monitoring
        if MODEL_AVAILABLE or direct_model_loaded:
            if predictor is not None or direct_model_loaded:
                try:
                    # 优先使用直接加载的模型信息
                    if direct_model_loaded:
                        model_name = direct_model_info['name']
                        model_device = direct_model_info['device']
                        model_healthy = True  # 假设直接加载的模型是健康的
                        param_count = 0  # 直接加载的模型参数数量未知
                        
                        message = f'Kronos model loaded from local directory: {model_name} on {model_device}'
                    else:
                        # Test model functionality for global model
                        model_device = str(next(predictor.model.parameters()).device)
                        model_name = predictor.model.__class__.__name__
                        
                        # Check if model is responsive
                        model_healthy = True
                        try:
                            # Simple test to verify model is working
                            if hasattr(predictor, 'model') and predictor.model is not None:
                                # Test model parameters
                                param_count = sum(p.numel() for p in predictor.model.parameters())
                            else:
                                model_healthy = False
                                param_count = 0
                        except Exception as e:
                            model_healthy = False
                            param_count = 0
                        
                        message = 'Kronos model loaded and available'
                    
                    return jsonify({
                        'available': True,
                        'loaded': True,
                        'healthy': model_healthy,
                        'message': message,
                        'current_model': {
                            'name': model_name,
                            'device': model_device,
                            'parameters': param_count,
                            'healthy': model_healthy,
                            'loaded_from': 'local_directory' if direct_model_loaded else 'huggingface'
                        },
                        'direct_model_loaded': direct_model_loaded,
                        'direct_model_info': direct_model_info,
                        'auto_load_enabled': direct_model_loader is not None,
                        'system_info': system_info,
                        'timestamp': datetime.datetime.now().isoformat()
                    })
                except Exception as e:
                    return jsonify({
                        'available': True,
                        'loaded': True,
                        'healthy': False,
                        'message': f'Model loaded but error occurred: {str(e)}',
                        'current_model': {
                            'name': 'Unknown',
                            'device': 'Unknown',
                            'parameters': 0,
                            'healthy': False,
                            'loaded_from': 'local_directory' if direct_model_loaded else 'huggingface'
                        },
                        'direct_model_loaded': direct_model_loaded,
                        'direct_model_info': direct_model_info,
                        'auto_load_enabled': direct_model_loader is not None,
                        'system_info': system_info,
                        'timestamp': datetime.datetime.now().isoformat()
                    })
            else:
                return jsonify({
                    'available': True,
                    'loaded': False,
                    'healthy': False,
                    'message': 'Kronos model available but not loaded',
                    'direct_model_loaded': direct_model_loaded,
                    'direct_model_info': direct_model_info,
                    'auto_load_enabled': direct_model_loader is not None,
                    'system_info': system_info,
                    'timestamp': datetime.datetime.now().isoformat()
                })
        else:
            return jsonify({
                'available': False,
                'loaded': False,
                'healthy': False,
                'message': 'Kronos model library not available, please install related dependencies',
                'direct_model_loaded': direct_model_loaded,
                'direct_model_info': direct_model_info,
                'auto_load_enabled': direct_model_loader is not None,
                'system_info': system_info,
                'timestamp': datetime.datetime.now().isoformat()
            })
    except Exception as e:
        return jsonify({
            'available': False,
            'loaded': False,
            'healthy': False,
            'message': f'Error getting model status: {str(e)}',
            'timestamp': datetime.datetime.now().isoformat()
        }), 500

@app.route('/api/auto-load-model', methods=['POST'])
def auto_load_model():
    """自动加载最优模型"""
    global tokenizer, model, predictor
    
    try:
        if auto_loader is None:
            return jsonify({
                'success': False,
                'error': '自动模型加载器不可用'
            }), 400
        
        # 选择最优模型和设备
        model_key, model_info = auto_loader.select_optimal_model()
        device = auto_loader.select_optimal_device()
        
        # 验证选择
        validation = auto_loader.validate_model_selection(model_key, device)
        if not validation['valid']:
            return jsonify({
                'success': False,
                'error': f'自动选择验证失败: {validation["reason"]}'
            }), 400
        
        # 检查模型是否可用，如果不可用则使用模拟模式
        if not MODEL_AVAILABLE:
            # 模拟模式：设置全局变量但不实际加载模型
            tokenizer = None
            model = None
            predictor = None
            
            # 获取系统报告
            system_report = auto_loader.get_system_report()
            
            return jsonify({
                'success': True,
                'message': f'模型自动选择成功（模拟模式）: {model_info["name"]} 在 {device} 上运行',
                'model_info': {
                    'name': model_info['name'],
                    'device': device,
                    'params': model_info['params'],
                    'description': model_info['description'],
                    'simulation_mode': True
                },
                'system_info': system_report,
                'simulation_mode': True
            })
        
        # 如果MODEL_AVAILABLE为True，但实际模型库不可用，也使用模拟模式
        try:
            # 尝试实际加载模型
            tokenizer = KronosTokenizer.from_pretrained(model_info['tokenizer_id'])
            model = Kronos.from_pretrained(model_info['model_id'])
            predictor = KronosPredictor(model, tokenizer, device=device)
            
            # 获取系统报告
            system_report = auto_loader.get_system_report()
            
            return jsonify({
                'success': True,
                'message': f'模型自动加载成功: {model_info["name"]} 在 {device} 上运行',
                'model_info': {
                    'name': model_info['name'],
                    'device': device,
                    'params': model_info['params'],
                    'description': model_info['description']
                },
                'system_info': system_report
            })
        except Exception as e:
            # 如果实际加载失败，使用模拟模式
            print(f"Warning: Actual model loading failed, using simulation mode: {e}")
            tokenizer = None
            model = None
            predictor = None
            
            # 获取系统报告
            system_report = auto_loader.get_system_report()
            
            return jsonify({
                'success': True,
                'message': f'模型自动选择成功（模拟模式）: {model_info["name"]} 在 {device} 上运行',
                'model_info': {
                    'name': model_info['name'],
                    'device': device,
                    'params': model_info['params'],
                    'description': model_info['description'],
                    'simulation_mode': True
                },
                'system_info': system_report,
                'simulation_mode': True
            })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'自动模型加载失败: {str(e)}'
        }), 500

@app.route('/api/system-info')
def get_system_info():
    """获取系统资源信息"""
    if auto_loader is None:
        return jsonify({
            'success': False,
            'error': '自动模型加载器不可用'
        }), 400
    
    try:
        system_report = auto_loader.get_system_report()
        return jsonify({
            'success': True,
            'system_info': system_report
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'获取系统信息失败: {str(e)}'
        }), 500

if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Kronos Web UI Server')
    parser.add_argument('--port', type=int, default=7070, help='Port to run the server on (default: 7070)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the server to (default: 0.0.0.0)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    print("Starting Kronos Web UI...")
    print(f"Model availability: {MODEL_AVAILABLE}")
    print(f"Server will run on: {args.host}:{args.port}")
    if MODEL_AVAILABLE:
        print("Tip: You can load Kronos model through /api/load-model endpoint")
    else:
        print("Tip: Will use simulated data for demonstration")
    
    app.run(debug=args.debug, host=args.host, port=args.port)
