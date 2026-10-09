"""Stub torch: list-backed tensors with the few operations eval.py uses."""

bfloat16 = "bfloat16"
long = "long"
__version__ = "0+mock"


class Tensor:
    def __init__(self, data):
        self.data = data

    def to(self, device):
        return self

    @property
    def shape(self):
        shape, node = [], self.data
        while isinstance(node, list):
            shape.append(len(node))
            node = node[0] if node else None
        return tuple(shape)

    def __getitem__(self, index):
        picked = self.data[index]
        return Tensor(picked) if isinstance(picked, list) else picked

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def tolist(self):
        return self.data


def tensor(data):
    return Tensor(data)


def ones(*shape, dtype=None):
    def build(dims):
        return [build(dims[1:]) for _ in range(dims[0])] if len(dims) > 1 else [1] * dims[0]

    return Tensor(build(list(shape)))


class no_grad:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class cuda:
    @staticmethod
    def is_available():
        return False
