from flax import nnx

class CriticTD3(nnx.Module):
    def __init__(self, rngs, in_dim):
        self.l1 = nnx.Linear(in_dim, 256, rngs=rngs)
        self.l2 = nnx.Linear(256, 256, rngs=rngs)
        self.l3 = nnx.Linear(256, 1, rngs=rngs)
    def __call__(self, x):
        return self.l3(nnx.relu(self.l2(nnx.relu(self.l1(x)))))

class ActorTD3(nnx.Module):
    def __init__(self, rngs, in_dim, out_dim, action_max):
        self.a_max = action_max
        self.l1 = nnx.Linear(in_dim, 256, rngs=rngs)
        self.l2 = nnx.Linear(256, 256, rngs=rngs)
        self.l3 = nnx.Linear(256, out_dim, rngs=rngs)
    def __call__(self, x):
        x = self.l3(nnx.relu(self.l2(nnx.relu(self.l1(x)))))
        return self.a_max * nnx.tanh(x)

# In this file I put the methods that produce the network parameters for the TD3 algorithm
def generate_value_network(rng, state_dim, action_dim):
    return CriticTD3(rng, state_dim)

def generate_policy_network(rng, state_dim, action_dim, action_max):
    return ActorTD3(rng, state_dim, action_dim, action_max)