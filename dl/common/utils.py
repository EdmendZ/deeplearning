import torch


def get_device():
    """自动检测并返回当前可用的计算设备。

    优先级：NVIDIA GPU(cuda) > Apple Silicon GPU(mps) > CPU。
    训练时可用 ``model.to(get_device())`` 把模型和数据迁移到该设备，
    无需手动判断硬件，保证代码在不同机器上都能跑。

    Returns:
        str: "cuda" / "mps" / "cpu" 三者之一。
    """
    if torch.cuda.is_available():          # 有 NVIDIA 显卡且 CUDA 可用
        return "cuda"
    elif torch.backends.mps.is_available():  # macOS 上的 Apple Silicon GPU
        return "mps"
    else:                                   # 兜底，回退到 CPU
        return "cpu"
