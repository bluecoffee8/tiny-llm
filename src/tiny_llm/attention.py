import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    factor = mx.rsqrt(query.shape[-1]) if scale is None else scale 
    scores = mx.matmul(query, key.swapaxes(-1, -2)) * factor
    if mask is not None:
        scores = scores + mask 
    return mx.matmul(softmax(scores, axis=-1), value)

class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.head_size = hidden_size // num_heads
        self.num_heads = num_heads
        self.scale = mx.rsqrt(self.head_size)
        self.wq = wq
        self.wk = wk 
        self.wv = wv
        self.wo = wo 

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        N, L, _ = query.shape 
        projection_q = (
            linear(query, self.wq)
            .reshape(N, L, self.num_heads, self.head_size)
            .transpose(0, 2, 1, 3)
        )
        projection_k = (
            linear(key, self.wk)
            .reshape(N, L, self.num_heads, self.head_size)
            .transpose(0, 2, 1, 3)
        )
        projection_v = (
            linear(value, self.wv)
            .reshape(N, L, self.num_heads, self.head_size)
            .transpose(0, 2, 1, 3)
        )
        scores = mx.matmul(projection_q, projection_k.swapaxes(-1, -2)) * self.scale 
        if mask is not None:
            scores = scores + mask 
        output = mx.matmul(softmax(scores, axis=-1), projection_v)
        return linear(output.transpose(0, 2, 1, 3).reshape(N, L, self.num_heads * self.head_size), self.wo)


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    pass


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    pass
