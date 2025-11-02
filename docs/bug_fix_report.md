# Bug修复报告：修复日期参数格式问题导致的400错误

## 问题描述
当用户在前端界面输入无效的日期格式时，系统会返回400错误，但错误信息不够明确，用户体验不佳。

## 根本原因分析
通过代码审查发现，问题出在`app.py`文件中处理`start_date`参数的部分。代码直接使用`pd.to_datetime(start_date)`转换用户输入的日期字符串，但没有对无效格式进行捕获和处理，导致抛出ValueError异常，进而返回400错误。

## 修复方案
在`app.py`文件中所有处理`start_date`参数的地方添加日期格式验证，使用try-except块捕获`pd.to_datetime()`可能抛出的ValueError异常，并返回明确的错误信息。

## 修复实施
在`app.py`文件中找到4处处理`start_date`参数的代码位置，并为它们添加了日期格式验证：

1. 第一处（约第965行）：
```python
# 修复前
start_dt = pd.to_datetime(start_date)

# 修复后
try:
    start_dt = pd.to_datetime(start_date)
except ValueError:
    return jsonify({'error': f'无效的开始日期格式: {start_date}，请使用 YYYY-MM-DD 格式'}), 400
```

2. 第二处（约第1059行）：
```python
# 修复前
start_dt = pd.to_datetime(start_date)

# 修复后
try:
    start_dt = pd.to_datetime(start_date)
except ValueError:
    return jsonify({'error': f'无效的开始日期格式: {start_date}，请使用 YYYY-MM-DD 格式'}), 400
```

3. 第三处（约第1159行）：
```python
# 修复前
start_dt = pd.to_datetime(start_date)

# 修复后
try:
    start_dt = pd.to_datetime(start_date)
except ValueError:
    return jsonify({'error': f'无效的开始日期格式: {start_date}，请使用 YYYY-MM-DD 格式'}), 400
```

4. 第四处（约第1175行）：
```python
# 修复前
start_dt = pd.to_datetime(start_date)

# 修复后
try:
    start_dt = pd.to_datetime(start_date)
except ValueError:
    return jsonify({'error': f'无效的开始日期格式: {start_date}，请使用 YYYY-MM-DD 格式'}), 400
```

## 验证结果
通过API测试验证修复效果：

1. 使用无效日期格式测试：
   ```bash
   curl -X POST http://localhost:7070/api/predict -H "Content-Type: application/json" -d '{"file_path":"stock_600159_live","start_date":"invalid-date"}'
   ```
   返回结果：
   ```json
   {"error":"无效的开始日期格式: invalid-date，请使用 YYYY-MM-DD 格式"}
   ```

2. 使用有效日期格式测试：
   ```bash
   curl -X POST http://localhost:7070/api/predict -H "Content-Type: application/json" -d '{"file_path":"stock_600159_live","start_date":"2023-01-01"}'
   ```
   返回结果：200状态码和预测数据

## 结论
修复成功解决了日期参数格式问题导致的400错误。现在当用户输入无效的日期格式时，系统会返回明确的错误信息，指导用户使用正确的日期格式（YYYY-MM-DD），提升了用户体验。