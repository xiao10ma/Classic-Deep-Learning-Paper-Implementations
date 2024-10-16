import math
import torch
from torch import nn

class PrepareForMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        """
        初始化多头注意力的准备层。

        参数:
        d_model: int, 输入的嵌入维度
        heads: int, 注意力头的数量
        d_k: int, 每个头的键（key）和查询（query）的维度
        bias: bool, 是否在线性变换中使用偏置
        """
        super().__init__()
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        self.heads = heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor):
        """
        将输入张量转换为多头格式。

        参数:
        x: torch.Tensor, 输入张量，形状为 (..., d_model)

        返回:
        torch.Tensor, 转换后的张量，形状为 (..., heads, d_k)
        """
        head_shape = x.shape[:-1]
        x = self.linear(x)
        x = x.view(*head_shape, self.heads, self.d_k)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1, bias: bool = True):
        super().__init__()
        self.d_k = d_model // heads
        self.heads = heads
        # Q, K, V
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)

        self.softmax = nn.Softmax(dim=1)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_prob)
        self.scale = 1 / math.sqrt(self.d_k)

        self.attn = None

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        return torch.