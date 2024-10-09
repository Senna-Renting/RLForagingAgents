from flax import nnx
import jax.numpy as jnp

class CriticTD3(nnx.Module):
    def __init__(self, rngs, in_dim, hidden_dim=256):
        self.l1 = nnx.Linear(in_dim, hidden_dim, rngs=rngs)
        self.l2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.l3 = nnx.Linear(hidden_dim, 1, rngs=rngs)
    def __call__(self, state, action):
        x = jnp.concatenate([state, action], axis=-1)
        return self.l3(nnx.elu(self.l2(nnx.elu(self.l1(x)))))

class ActorTD3(nnx.Module):
    def __init__(self, rngs, in_dim, out_dim, action_max, hidden_dim=256):
        self.a_max = action_max
        self.l1 = nnx.Linear(in_dim, hidden_dim, rngs=rngs)
        self.l2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.l3 = nnx.Linear(hidden_dim, out_dim, rngs=rngs)
    def __call__(self, state):
        x = self.l3(nnx.elu(self.l2(nnx.elu(self.l1(state)))))
        return self.a_max * nnx.tanh(x)

# In this file I put the methods that produce the network parameters for the TD3 algorithm
def generate_value_network(rng, state_dim, action_dim):
    return CriticTD3(rng, state_dim+action_dim)

def generate_policy_network(rng, state_dim, action_dim, action_max):
    return ActorTD3(rng, state_dim, action_dim, action_max)