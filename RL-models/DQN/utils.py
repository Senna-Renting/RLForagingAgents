from collections import deque
from jax import random
import jax.numpy as jnp

class ReplayBuffer:
    def __init__(self, maxlen, seed=0):
        self.max_len = maxlen
        self.buf = deque(maxlen=maxlen)
        self.key = random.PRNGKey(seed)

    def sample_batch(self, batch_size):
        self.key = random.split(self.key)[0]
        idxs = random.randint(self.key, (batch_size,), 0, len(self))
        batch = [self[idx] for idx in idxs]

        # Each item to be its own tensor of len batch_size
        b = list(zip(*batch))
        buf_mean = 0
        buf_std = 1
        return [(jnp.asarray(t) - buf_mean) / buf_std for t in b]

    def append(self, x):
        self.buf.append(x)

    def __getitem__(self, idx):
        return self.buf[idx]

    def __len__(self):
        return len(self.buf)