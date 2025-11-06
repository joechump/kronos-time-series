#!/usr/bin/env python3
"""
测试PyTorchModelHubMixin的from_pretrained方法要求
"""

from huggingface_hub import PyTorchModelHubMixin
import inspect

# 检查PyTorchModelHubMixin的from_pretrained方法
print("=== PyTorchModelHubMixin from_pretrained方法检查 ===")

# 获取from_pretrained方法的签名
if hasattr(PyTorchModelHubMixin, 'from_pretrained'):
    method = getattr(PyTorchModelHubMixin, 'from_pretrained')
    print(f"from_pretrained方法存在: {method}")
    
    # 获取方法签名
    sig = inspect.signature(method)
    print(f"方法签名: {sig}")
    
    # 获取参数信息
    print("参数详情:")
    for param_name, param in sig.parameters.items():
        print(f"  {param_name}: {param.annotation} = {param.default}")
else:
    print("from_pretrained方法不存在")

# 检查PyTorchModelHubMixin的文档
print("\n=== PyTorchModelHubMixin文档检查 ===")
print(f"文档: {PyTorchModelHubMixin.__doc__}")

# 检查是否有其他相关方法
print("\n=== PyTorchModelHubMixin其他方法检查 ===")
methods = [method for method in dir(PyTorchModelHubMixin) if not method.startswith('_')]
print(f"公共方法: {methods}")

# 检查save_pretrained方法
if hasattr(PyTorchModelHubMixin, 'save_pretrained'):
    method = getattr(PyTorchModelHubMixin, 'save_pretrained')
    sig = inspect.signature(method)
    print(f"save_pretrained方法签名: {sig}")

print("\n=== 测试完成 ===")