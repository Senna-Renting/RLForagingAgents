from flax import nnx
import jax.numpy as jnp
import jax
from functools import partial
import gymnax
import gymnasium as gym
import numpy as np
import optax

## Critic network and it's complementary functions
class Critic(nnx.Module):
    def __init__(self, in_dim, rngs: nnx.Rngs, hidden_dim=256):
        self.l1 = nnx.Linear(in_dim, hidden_dim, rngs=rngs)
        self.l2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.l3 = nnx.Linear(hidden_dim, 1, rngs=rngs)
    def __call__(self, state, action):
        x = jnp.concatenate([state, action], axis=-1)
        return self.l3(nnx.relu(self.l2(nnx.relu(self.l1(x)))))

@nnx.jit
def MSE_optimize_critic(optimizer: nnx.Optimizer, critic: nnx.Module, states: jnp.array, actions: jnp.array, ys: jnp.array):
    loss_fn = lambda critic: ((critic(states, actions) - ys) ** 2).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(critic)
    optimizer.update(grads)
    return loss, grads

@nnx.jit
def compute_targets(critic: nnx.Module, actor: nnx.Module, rs: jnp.array, states: jnp.array, done: jnp.array, gamma: float):
    return rs + gamma*(1-done)*(critic(states, actor(states)))

## Actor network and it's complementary functions
class Actor(nnx.Module):
    def __init__(self, in_dim, out_dim, action_max, rngs: nnx.Rngs, hidden_dim=256):
        self.a_max = action_max
        self.l1 = nnx.Linear(in_dim, hidden_dim, rngs=rngs)
        self.l2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.l3 = nnx.Linear(hidden_dim, out_dim, rngs=rngs)
    def __call__(self, state):
        x = self.l3(nnx.relu(self.l2(nnx.relu(self.l1(state)))))
        return self.a_max * nnx.tanh(x)

@nnx.jit
def mean_optimize_actor(optimizer: nnx.Optimizer, actor: nnx.Module, critic: nnx.Module, states: jnp.array):
    loss_fn = lambda actor: -1*critic(states, actor(states)).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(actor)
    optimizer.update(grads)
    return loss, grads

def sample_action(rng, actor, state, action_min, action_max):
    mu_action = actor(state)
    eps = jax.random.normal(rng, (1,))
    return jnp.clip(mu_action + eps, action_min, action_max)

@nnx.jit
def polyak_update(tau: float, net_target: nnx.Module, net_normal: nnx.Module):
    params_t = nnx.state(net_target)
    params_n = nnx.state(net_normal)
    update_fn = lambda param_t, param_n: tau*param_n + (1-tau)*param_t
    param_new = jax.tree.map(update_fn, params_t, params_n)
    return param_new
    

## Buffer data structure
# Note: numpy is used in this structure as I need to dynamically change the buffer over time
# Implications: not JIT-compileable structure, but the output of the buffer when sampling does
# contain jax arrays, so from that point onward we should be able to JIT.
class Buffer:
    def __init__(self, buffer_size, state_dim, action_dim):
        self.states = np.empty((buffer_size,state_dim))
        self.actions = np.empty((buffer_size,action_dim))
        self.rewards = np.empty((buffer_size,1))
        self.next_states = np.empty((buffer_size,state_dim))
        self.dones = np.empty((buffer_size,1))
        self.max_size = buffer_size
        self.size = 0
        self.ptr = 0
        self.rng = np.random.default_rng(0)
    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr, :] = state
        self.actions[self.ptr, :] = action
        self.rewards[self.ptr, :] = reward
        self.next_states[self.ptr, :] = next_state
        self.dones[self.ptr, :] = done
        self.size = min(self.size + 1, self.max_size)
        self.ptr = (self.ptr + 1) % self.max_size
    def sample(self, batch_size):
        ind = self.rng.integers(0,self.size,size=batch_size)
        return (
            jax.device_put(self.states[ind]),
            jax.device_put(self.actions[ind]),
            jax.device_put(self.rewards[ind]),
            jax.device_put(self.next_states[ind]),
            jax.device_put(self.dones[ind]),
        ) 

## Train loop of DDPG algorithm
def train_ddpg(num_episodes, tau=0.05, gamma=0.99, batch_size=256, buffer_size=10000, lr=1e-3, seed=0):
    # Initialize neural networks
    action_max = 2
    state_dim = 3
    action_dim = 1
    reset_key = 43
    actor = Actor(state_dim,action_dim,action_max,nnx.Rngs(0),hidden_dim=32)
    actor_t = Actor(state_dim,action_dim,action_max,nnx.Rngs(0),hidden_dim=32)
    critic = Critic(state_dim + action_dim, nnx.Rngs(1),hidden_dim=32)
    critic_t = Critic(state_dim + action_dim, nnx.Rngs(1),hidden_dim=32)
    optim_actor = nnx.Optimizer(actor, optax.adam(lr))
    optim_critic = nnx.Optimizer(critic, optax.adam(lr))
    # Add buffer
    buffer = Buffer(buffer_size, state_dim, action_dim)
    # Initialize environment
    key = jax.random.PRNGKey(40)
    env = gym.make("Pendulum-v1")
    # Keep track of accumulated rewards
    returns = np.zeros(num_episodes)
    for i in range(num_episodes):
        done = False
        state, info = env.reset(seed=reset_key)
        while not done:
            # Sample action, execute it, and add to buffer
            action_key, key = jax.random.split(key)
            action = sample_action(action_key, actor, state, -action_max, action_max)
            next_state, reward, terminated, truncated, _ = env.step(action)
            returns[i] += reward
            done = truncated or terminated
            buffer.add(state, action, reward, next_state, terminated)
            state = next_state
            # Sample batch from buffer
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
            # Update critic
            ys = compute_targets(critic_t, actor, rewards, next_states, dones, gamma)
            c_loss, grads = MSE_optimize_critic(optim_critic, critic, states, actions, ys)
            # Update policy
            a_loss, grads = mean_optimize_actor(optim_actor, actor, critic, states)
            # Update targets (critic and policy)
            nnx.update(critic_t, polyak_update(tau, critic_t, critic))
            nnx.update(actor_t, polyak_update(tau, actor_t, actor))
        print(f"Episode {i} done")
        print(f"Accumulated rewards: {returns[i]}")
    return returns, actor_t, critic_t, reset_key
            