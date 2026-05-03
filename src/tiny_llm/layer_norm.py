import mlx.core as mx


class RMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        self.dim = dim 
        self.weight = weight 
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.multiply(x, mx.rsqrt(mx.sum(mx.square(x.astype(mx.float32)), axis=-1).reshape(*x.shape[:-1], 1) / self.dim + self.eps)) * self.weight
