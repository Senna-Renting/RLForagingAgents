import jax.numpy as jnp
from jax import random
from flax import nnx
from buffer import ReplayBuffer
from parameters import generate_value_network, generate_policy_network

class TD3:
    def __init__(self, seed, state_dim, action_dim, 
                 buffer_size, action_max, policy_interval=10,
                 tau=0.8, sigma=1):
        self.buffer = ReplayBuffer(state_dim, action_dim, buffer_size)
        self.action_max = action_max
        self.policy_interval = policy_interval
        self.tau = tau
        self.sigma = sigma
        self.params_rng = nnx.Rngs(seed)
        self.action_rng = random.PRNGKey(seed)
        self.Qnet1 = generate_value_network(self.params_rng, state_dim, action_dim)
        self.Qnet2 = generate_value_network(self.params_rng, state_dim, action_dim)
        self.Pnet = generate_policy_network(self.params_rng, state_dim, action_dim, action_max)
        self.Qnet1_target = generate_value_network(self.params_rng, state_dim, action_dim)
        self.Qnet2_target = generate_value_network(self.params_rng, state_dim, action_dim)
        self.Pnet_target = generate_policy_network(self.params_rng, state_dim, action_dim, action_max)
        # Make sure targets are equal in state w.r.t their no-target versions
        _, state = nnx.split(self.Qnet1)
        nnx.update(self.Qnet1_target,state)
        _, state = nnx.split(self.Qnet2)
        nnx.update(self.Qnet2_target,state)
        _, state = nnx.split(self.Pnet)
        nnx.update(self.Pnet_target,state)

    def select_action(self, state):
        return self.Pnet_target(state)

    def sample_action(self, state):
        mu = self.select_action(state)
        self.action_rng, subkey = random.split(self.action_rng)
        action = mu + self.action_max * self.sigma * random.normal(subkey, mu.shape)
        return jnp.clip(action, -self.action_max, self.action_max)

    def train(self):
        pass

if __name__ == "__main__":
    rngs = nnx.Rngs(0)
    model = TD3(0,3,1,1000,3)
    print(model.sample_action(jnp.array([1,2,3])))
        