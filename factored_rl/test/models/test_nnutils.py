import pytest

import torch

from factored_rl.models.nnutils import one_hot

def test_one_hot_single():
    N = 5
    x = torch.tensor(N - 1)
    y = torch.zeros(N)
    y[-1] = 1
    assert torch.equal(one_hot(x, N), y)

def test_one_hot_vector():
    N = 5
    x = torch.arange(N)
    y = torch.eye(N)
    assert torch.equal(one_hot(x, N), y)

def test_one_hot_matrix():
    N = 5
    batch_size = 10
    x = torch.arange(N).unsqueeze(0).expand(batch_size, -1)
    y = torch.eye(N).unsqueeze(0).expand(batch_size, -1, -1)
    assert torch.equal(one_hot(x, N), y)

def test_one_hot_tensor():
    N = 3
    x = (torch.arange(24) % N).reshape(2, 3, 4)
    y = torch.eye(N).reshape(1, 3, 3).expand(24 // N, N, N).reshape(2, 3, 4, 3)
    assert torch.equal(one_hot(x, N), y)

def test_one_hot_error():
    N = 5
    x = torch.tensor(N + 2)
    with pytest.raises(RuntimeError):
        one_hot(x, N)
