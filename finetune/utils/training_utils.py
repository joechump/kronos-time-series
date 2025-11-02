import os
import random
import datetime
import numpy as np
import torch
import torch.distributed as dist


def setup_ddp():
    """
    初始化分布式数据并行环境。

    此函数依赖于`torchrun`或类似启动器设置的环境变量。
    它初始化进程组并为当前进程设置CUDA设备。

    返回:
        tuple: 包含(rank, world_size, local_rank)的元组。
    """
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available.")

    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    print(
        f"[DDP Setup] Global Rank: {rank}/{world_size}, "
        f"Local Rank (GPU): {local_rank} on device {torch.cuda.current_device()}"
    )
    return rank, world_size, local_rank


def cleanup_ddp():
    """清理分布式进程组。"""
    if dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int, rank: int = 0):
    """
    为所有相关库设置随机种子以确保可重现性。

    参数:
        seed (int): 基础种子值。
        rank (int): 进程排名，用于确保不同进程具有不同的种子，
                    这对于数据加载可能很重要。
    """
    actual_seed = seed + rank
    random.seed(actual_seed)
    np.random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual_seed)
        # The two lines below can impact performance, so they are often
        # reserved for final experiments where reproducibility is critical.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_model_size(model: torch.nn.Module) -> str:
    """
    计算PyTorch模型中可训练参数的数量，并以人类可读的字符串形式返回。

    参数:
        model (torch.nn.Module): PyTorch模型。

    返回:
        str: 表示模型大小的字符串（例如"175.0B"、"7.1M"、"50.5K"）。
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if total_params >= 1e9:
        return f"{total_params / 1e9:.1f}B"  # Billions
    elif total_params >= 1e6:
        return f"{total_params / 1e6:.1f}M"  # Millions
    else:
        return f"{total_params / 1e3:.1f}K"  # Thousands


def reduce_tensor(tensor: torch.Tensor, world_size: int, op=dist.ReduceOp.SUM) -> torch.Tensor:
    """
    在分布式设置中跨所有进程减少张量的值。

    参数:
        tensor (torch.Tensor): 要减少的张量。
        world_size (int): 进程总数。
        op (dist.ReduceOp, 可选): 减少操作（SUM、AVG等）。
                                  默认为dist.ReduceOp.SUM。

    返回:
        torch.Tensor: 减少后的张量，在所有进程上将是相同的。
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=op)
    # Note: `dist.ReduceOp.AVG` is available in newer torch versions.
    # For compatibility, manual division is sometimes used after a SUM.
    if op == dist.ReduceOp.AVG:
        rt /= world_size
    return rt


def format_time(seconds: float) -> str:
    """
    将秒数格式化为人类可读的H:M:S字符串。

    参数:
        seconds (float): 总秒数。

    返回:
        str: 格式化后的时间字符串（例如"0:15:32"）。
    """
    return str(datetime.timedelta(seconds=int(seconds)))



