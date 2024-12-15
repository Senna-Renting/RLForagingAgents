from flax import nnx
import jax.numpy as jnp
import jax
import gymnax
import gymnasium as gym
import numpy as np
import optax
import wandb
from environment import *
from save_utils import save_policy

## Critic network and it's complementary functions
class Critic(nnx.Module):
    def __init__(self, in_dim, seed, hidden_dim=32):
        self.l1 = nnx.Linear(in_dim, hidden_dim, rngs=nnx.Rngs(seed))
        self.l2 = nnx.Linear(hidden_dim, hidden_dim, rngs=nnx.Rngs(seed+1))
        self.l3 = nnx.Linear(hidden_dim, 1, rngs=nnx.Rngs(seed+2))
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
    def __init__(self, in_dim, out_dim, action_max, seed, hidden_dim=32):
        self.a_max = action_max
        self.l1 = nnx.Linear(in_dim, hidden_dim, rngs=nnx.Rngs(seed))
        self.l2 = nnx.Linear(hidden_dim, hidden_dim, rngs=nnx.Rngs(seed+1))
        self.l3 = nnx.Linear(hidden_dim, out_dim, rngs=nnx.Rngs(seed+2))
    def __call__(self, state):
        x = self.l3(nnx.relu(self.l2(nnx.relu(self.l1(state)))))
        return self.a_max * nnx.tanh(x)

@nnx.jit
def mean_optimize_actor(optimizer: nnx.Optimizer, actor: nnx.Module, critic: nnx.Module, states: jnp.array):
    loss_fn = lambda actor: -1*critic(states, actor(states)).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(actor)
    optimizer.update(grads)
    return loss, grads

def sample_action(rng, actor, state, action_min, action_max, action_dim):
    mu_action = actor(state)
    eps = jax.random.normal(rng, (action_dim,))
    #print("Action (no noise): ", mu_action, "; Action (w/ noise): ", mu_action + eps)
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

# Below I build a curried function designed to work with wandb
def wandb_train_ddpg(env):
    def do_wandb(config=None):
        with wandb.init(config=config) as run:
            config = wandb.config
            print(config)
            returns, actor_t, *_ = train_ddpg(env, log_fun=wandb_log_ddpg, **config)
            # Save policy of ddpg algorithm
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policies", run.sweep_id, run.name)
            save_policy(actor_t, path)
    return do_wandb

def print_log_ddpg(epoch, critic_loss, actor_loss, returns):
    print(f"Episode {epoch} done")
    print(f"Critic loss: {critic_loss}")
    print(f"Actor loss: {actor_loss}")
    print(f"Return: {returns}")

def wandb_log_ddpg(epoch, critic_loss, actor_loss, returns):
    wandb.log({"Epoch":epoch,
               "Critic loss":critic_loss,
               "Actor loss":actor_loss,
               "Return":returns})

## Train loop of DDPG algorithm
def train_ddpg(env, num_episodes, tau=0.05, gamma=0.99, batch_size=64, lr_a=1e-4, lr_c=3e-4, seed=0, reset_seed=43, action_dim=1, state_dim=3, action_max=2, hidden_dim=256, warmup_steps=800, log_fun=print_log_ddpg):
    # Initialize neural networks
    actor = Actor(state_dim,action_dim,action_max,seed,hidden_dim=hidden_dim)
    actor_t = Actor(state_dim,action_dim,action_max,seed,hidden_dim=hidden_dim)
    critic = Critic(state_dim + action_dim,seed,hidden_dim=hidden_dim)
    critic_t = Critic(state_dim + action_dim,seed,hidden_dim=hidden_dim)
    optim_actor = nnx.Optimizer(actor, optax.adam(lr_a))
    optim_critic = nnx.Optimizer(critic, optax.adam(lr_c))
    # Add buffer
    buffer_size = num_episodes*env.step_max+warmup_steps # Ensure every step is kept in the replay buffer
    buffer = Buffer(buffer_size, state_dim, action_dim)
    # Initialize environment
    key = jax.random.PRNGKey(seed)
    # Keep track of accumulated rewards
    returns = np.zeros(num_episodes)
    # Warm-up the buffer
    np.random.seed(reset_seed)
    rand_actions = np.random.uniform(low=-action_max, high=action_max, size=(warmup_steps,action_dim))
    state, info = env.reset(seed=reset_seed)
    for j in range(rand_actions.shape[0]):
        action = rand_actions[j]
        next_state, reward, terminated, truncated, _ = env.step(jnp.array(action))
        buffer.add(state, action, reward, next_state, terminated)
        state = next_state
    # Run episodes
    for i in range(num_episodes):
        done = False
        state, info = env.reset(seed=reset_seed)
        # Train agent
        while not done:
            # Sample action, execute it, and add to buffer
            action_key, key = jax.random.split(key)
            action = sample_action(action_key, actor, state, -action_max, action_max, action_dim)
            next_state, reward, terminated, truncated, _ = env.step(jnp.array(action))
            #print("Action: ", action)
            #print(reward)
            #print("State: ", next_state)
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
        # Test agent
        done = False
        state, info = env.reset(seed=reset_seed)
        while not done:
            action = actor_t(state)
            next_state, reward, terminated, truncated, _ = env.step(jnp.array(action))
            state = next_state
            done = terminated or truncated
            returns[i] += reward
        # Log the important variables to some logger
        log_fun(i, c_loss, a_loss, returns[i])
    return returns, actor_t, critic_t, reset_seed
            