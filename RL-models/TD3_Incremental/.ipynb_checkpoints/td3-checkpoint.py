from flax import nnx
import jax.numpy as jnp
import optax

class Critic(nnx.Module):
    def __init__(self, in_dim, rngs: nnx.Rngs, hidden_dim=256):
        self.l1 = nnx.Linear(in_dim, hidden_dim, rngs=rngs)
        self.l2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.l3 = nnx.Linear(hidden_dim, 1, rngs=rngs)
    def __call__(self, state, action):
        x = jnp.concatenate([state, action], axis=-1)
        return self.l3(nnx.relu(self.l2(nnx.relu(self.l1(x)))))

def MSE_optimize_critic(optimizer: nnx.Optimizer, critic: nnx.Module, states: jnp.array, actions: jnp.array, ys: jnp.array):
    loss_fn = lambda critic: ((critic(states, actions) - ys) ** 2).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(critic)
    optimizer.update(grads)
    return loss, grads

def compute_targets(critic: nnx.Module, actor: nnx.Module, rs: jnp.array, states: jnp.array, done: jnp.array, gamma: float):
    return rs + gamma*(1-done)*(critic(states, actor(states)))

class Actor(nnx.Module):
    def __init__(self, in_dim, out_dim, action_max, rngs: nnx.Rngs, hidden_dim=256):
        self.a_max = action_max
        self.l1 = nnx.Linear(in_dim, hidden_dim, rngs=rngs)
        self.l2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.l3 = nnx.Linear(hidden_dim, out_dim, rngs=rngs)
    def __call__(self, state):
        x = self.l3(nnx.relu(self.l2(nnx.relu(self.l1(state)))))
        return self.a_max * nnx.tanh(x)

def mean_optimize_actor(optimizer: nnx.Optimizer, actor: nnx.Module, critic: nnx.Module, states: jnp.array):
    loss_fn = lambda actor: -1*critic(states, actor(states)).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(actor)
    optimizer.update(grads)
    return loss, grads