from flax import nnx
import jax.numpy as jnp
import jax
import numpy as np
import optax
import wandb
from environment import *
from save_utils import save_policy
from functools import partial

# TODO: Implement the four new functions added, allowing multi-objective RL updates:
# 1. compute_NSW
# 2. compute_welfare_targets
# 3. optimize_welfare_critic
# 4. optimize_welfare_actor


## Critic network and it's complementary functions
class Critic(nnx.Module):
    def __init__(self, in_dim, seed, out_dim=1, hidden_dim=[16,32,16]):
        rngs = nnx.Rngs(seed)
        self.input = nnx.Linear(in_dim, hidden_dim[0], rngs=rngs)
        self.lhs = tuple([nnx.Linear(hidden_dim[i], hidden_dim[i+1], rngs=rngs) for i in range(len(hidden_dim) - 1)])
        self.out = nnx.Linear(hidden_dim[-1], out_dim, rngs=rngs)
    def __call__(self, state, action):
        x = jnp.concatenate([state, action], axis=-1)
        x = nnx.relu(self.input(x))
        for lh in self.lhs:
            x = nnx.leaky_relu(lh(x))
        return self.out(x)

@nnx.jit
def MSE_optimize_critic(optimizer: nnx.Optimizer, critic: nnx.Module, states: jnp.array, actions: jnp.array, ys: jnp.array):
    loss_fn = lambda critic: ((critic(states, actions) - ys) ** 2).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(critic)
    optimizer.update(grads)
    return loss, grads

@nnx.jit
def optimize_welfare_critic(optimizer: nnx.Optimizer, critic: nnx.Module, states: jnp.array, actions: jnp.array, ys: jnp.array):
    pass

@nnx.jit
def compute_targets(critic: nnx.Module, actor: nnx.Module, rs: jnp.array, states: jnp.array, done: jnp.array, gamma: float):
    return rs + gamma*(1-done)*(critic(states, actor(states)))

@nnx.jit
def compute_welfare_targets(critic: nnx.Module, actor: nnx.Module, rs: jnp.array, states: jnp.array, done: jnp.array, gamma: float):
    pass

## Actor network and it's complementary functions
class Actor(nnx.Module):
    def __init__(self, in_dim, out_dim, action_max, seed, hidden_dim=[16,32,16]):
        self.a_max = action_max
        rngs = nnx.Rngs(seed)
        self.input = nnx.Linear(in_dim, hidden_dim[0], rngs=rngs)
        self.lhs = tuple([nnx.Linear(hidden_dim[i], hidden_dim[i+1], rngs=rngs) for i in range(len(hidden_dim) - 1)])
        self.out = nnx.Linear(hidden_dim[-1], out_dim, rngs=rngs)
    def __call__(self, state):
        x = nnx.relu(self.input(state))
        for lh in self.lhs:
            x = nnx.leaky_relu(lh(x))
        x = self.out(x)
        return self.a_max * nnx.tanh(x)

@nnx.jit
def mean_optimize_actor(optimizer: nnx.Optimizer, actor: nnx.Module, critic: nnx.Module, states: jnp.array):
    loss_fn = lambda actor: -1*critic(states, actor(states)).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(actor)
    optimizer.update(grads)
    return loss, grads

@nnx.jit
def optimize_welfare_actor(optimizer: nnx.Optimizer, actor: nnx.Module, critic: nnx.Module, states: jnp.array):
    pass

def sample_action(rng, actor, state, action_min, action_max, action_dim):
    mu_action = actor(state)
    eps = jax.random.normal(rng, (action_dim,))
    return jnp.clip(mu_action + eps, action_min, action_max)

@nnx.jit
def polyak_update(tau: float, net_target: nnx.Module, net_normal: nnx.Module):
    params_t = nnx.state(net_target)
    params_n = nnx.state(net_normal)
    update_fn = lambda param_t, param_n: tau*param_n + (1-tau)*param_t
    param_new = jax.tree.map(update_fn, params_t, params_n)
    return param_new
    
# The function below returns a list of the weights of a nnx.Module (aka neural network) 
def get_network_weights(network):
    state = nnx.state(network)
    input = state["input"]["kernel"].value
    output = state["out"]["kernel"].value
    hidden = [state["lhs"][i]["kernel"].value for i in range(len(state["lhs"]))]
    return [input, *hidden, output]

# The function below returns a list of shapes required to store the nnx.Module's weights
def get_network_shape(network):
    state = nnx.state(network)
    input = state["input"]["kernel"].value.shape
    output = state["out"]["kernel"].value.shape
    hidden = [state["lhs"][i]["kernel"].value.shape for i in range(len(state["lhs"]))]
    return [input, *hidden, output]

# Helper function for computing the Nash Social Welfare function (aka geometric mean)
def compute_NSW(rewards):
    pass

## Buffer data structure
# Note: numpy is used in this structure as I need to dynamically change the buffer over time
# Implications: not JIT-compileable structure, but the output of the buffer when sampling does
# contain jax arrays, so from that point onward we should be able to JIT.
class Buffer:
    def __init__(self, buffer_size, state_dim, action_dim, reward_dim=1):
        self.states = np.empty((buffer_size,state_dim))
        self.actions = np.empty((buffer_size,action_dim))
        self.rewards = np.empty((buffer_size,reward_dim))
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
    def get(self, indices):
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )
    def get_all(self):
        return (
            self.states[np.newaxis, :],
            self.actions[np.newaxis, :],
            self.rewards[np.newaxis, :],
            self.next_states[np.newaxis, :],
            self.dones[np.newaxis, :],
        )
    def get_pointer(self):
        return self.ptr
    def sample(self, batch_size):
        ind = self.rng.integers(0,self.size,size=batch_size)
        return (
            jax.device_put(self.states[ind]),
            jax.device_put(self.actions[ind]),
            jax.device_put(self.rewards[ind]),
            jax.device_put(self.next_states[ind]),
            jax.device_put(self.dones[ind]),
        ), ind

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
    print(f"Return: {np.sum(returns, axis=1)}")

def print_log_ddpg_n_agents(epoch, returns, energies):
    print(f"Episode {epoch} done")
    [print(f"Agent {i+1}'s return: {np.sum(returns[:,i], axis=0)}") for i in range(returns.shape[1])]
    [print(f"Agent {i+1}'s energy: {energy}") for i,energy in enumerate(energies)]

def wandb_log_ddpg(epoch, critic_loss, actor_loss, returns):
    wandb.log({"Epoch":epoch,
               "Critic loss":critic_loss,
               "Actor loss":actor_loss,
               "Return":returns})

# TODO: Simplify this function by removing the welfare stuff (stuff relating to the p_welfare parameter)
def n_agents_ddpg(env, num_episodes, tau=0.01, gamma=0.99, batch_size=30, lr_a=2e-4, lr_c=8e-4, seed=0, action_dim=2, state_dim=3, action_max=0.2, hidden_dim=[16,16], log_fun=print_log_ddpg_n_agents):
    # Initialize metadata object for keeping track of (hyper-)parameters and/or additional settings of the environment
    hidden_dims = [str(h_dim) for h_dim in hidden_dim]
    warmup_size = 2*batch_size
    metadata = dict(n_episodes=num_episodes, tau=tau, gamma=gamma, 
                    batch_size=batch_size, lr_actor=lr_a, lr_critic=lr_c, 
                    seed=seed, action_dim=action_dim, state_dim=state_dim,
                    action_max=action_max, hidden_dims=hidden_dims,
                    warmup_size=warmup_size, alg_name="Normal DDPG", **env.get_params())
    # Initialize neural networks
    n_agents = env.n_agents
    step_max = env.step_max
    actor_dim = action_dim
    
    actors = [Actor(state_dim,actor_dim,action_max,seed+i,hidden_dim=hidden_dim) for i in range(n_agents)]
    actors_t = [Actor(state_dim,actor_dim,action_max,seed+i,hidden_dim=hidden_dim) for i in range(n_agents)]
    critics = [Critic(state_dim+actor_dim,seed+i,out_dim=1,hidden_dim=hidden_dim) for i in range(n_agents)]
    critics_t = [Critic(state_dim+actor_dim,seed+i,out_dim=1,hidden_dim=hidden_dim) for i in range(n_agents)]
    
    optim_actors = [nnx.Optimizer(actors[i], optax.adam(lr_a)) for i in range(n_agents)]
    optim_critics = [nnx.Optimizer(critics[i], optax.adam(lr_c)) for i in range(n_agents)]
    # Add seperate experience replay buffer for each agent 
    buffer_size = num_episodes*env.step_max+warmup_size # Ensure every step is kept in the replay buffer
    buffers = [Buffer(buffer_size, state_dim, actor_dim) for i in range(n_agents)]
    # Initialize environment
    key = jax.random.PRNGKey(seed)
    # Keep track of neural network weights (assumes homogeneous networks across agents!)
    critic_weights = [np.empty((num_episodes, n_agents, *shape)) for shape in get_network_shape(critics_t[0])]
    actor_weights = [np.empty((num_episodes, n_agents, *shape)) for shape in get_network_shape(actors_t[0])]
    # Keep track of important loss variables (3 is for [avg,min,max] stats)
    critics_loss_stats = np.empty((num_episodes, 3, n_agents))
    actors_loss_stats = np.empty((num_episodes, 3, n_agents))
    # Keep track of accumulated rewards
    returns = np.empty((num_episodes, step_max, n_agents, 1))
    test_penalties = np.empty((num_episodes, step_max, n_agents, 1))
    # Keep track of patch enter events and states of agents
    test_is_in_patch = np.empty((num_episodes, step_max, n_agents))
    (agents_state, patch_state, step_idx), states = env.reset(seed=seed)
    agent_states = np.empty((num_episodes, step_max, *agents_state.shape))
    patch_states = np.empty((num_episodes, step_max, 1))
    # Warm-up round
    env_state, states = env.reset(seed=seed)
    for s_i in range(warmup_size):
        # Sample action, execute it, and add to buffer
        actions = list(range(n_agents))
        for i_a,actor in enumerate(actors):
            action_key, key = jax.random.split(key)
            actions[i_a] = jnp.array(sample_action(action_key, actor, states[i_a], -action_max, action_max, actor_dim)) 
        env_state, next_states, (rewards, penalties), terminated, truncated, _ = env.step(env_state, *actions)
        (agents_state, patch_state, step_idx) = env_state
        for i_a in range(n_agents):
            buffers[i_a].add(states[i_a], actions[i_a], rewards[i_a], next_states[i_a], terminated)
        states = next_states
    # Run episodes
    for i in range(num_episodes):
        done = False
        env_state, states = env.reset(seed=seed)
        # Initialize loss temp variables
        cs_loss = np.empty((n_agents,step_max))
        as_loss = np.empty((n_agents,step_max))
        # Train agent
        for s_i in range(step_max):
            # Sample action, execute it, and add to buffer
            actions = list(range(n_agents))
            for i_a,actor in enumerate(actors):
                action_key, key = jax.random.split(key)
                actions[i_a] = jnp.array(sample_action(action_key, actor, states[i_a], -action_max, action_max, actor_dim)) 
            env_state, next_states, (rewards, penalties), terminated, truncated, _ = env.step(env_state, *actions)
            (agents_state, patch_state, step_idx) = env_state
            done = truncated or terminated
            if not terminated:
                for i_a in range(n_agents):
                    # Add states info to buffer
                    buffers[i_a].add(states[i_a], actions[i_a], rewards[i_a], next_states[i_a], terminated)
                    # Sample batch from buffer
                    (b_states, b_actions, b_rewards, b_next_states, b_dones), ind = buffers[i_a].sample(batch_size)
                    # Update critic
                    ys = compute_targets(critics_t[i_a], actors[i_a], b_rewards, b_next_states, b_dones, gamma)
                    c_loss, grads = MSE_optimize_critic(optim_critics[i_a], critics[i_a], b_states, b_actions, ys)
                    # Update policy
                    a_loss, grads = mean_optimize_actor(optim_actors[i_a], actors[i_a], critics[i_a], b_states)
                    # Update targets (critic and policy)
                    nnx.update(critics_t[i_a], polyak_update(tau, critics_t[i_a], critics[i_a]))
                    nnx.update(actors_t[i_a], polyak_update(tau, actors_t[i_a], actors[i_a]))
                    # Store actor and critic loss 
                    as_loss[i_a,step_idx-1] = a_loss
                    cs_loss[i_a,step_idx-1] = c_loss
            else:
                break
            # Update state of agents
            states = next_states
        # Save training results
        step_idx = env_state[2]
        actors_loss_stats[i,:,:] = [np.mean(as_loss, axis=1), np.min(as_loss, axis=1), np.max(as_loss, axis=1)]
        critics_loss_stats[i,:,:] = [np.mean(cs_loss, axis=1), np.min(cs_loss, axis=1), np.max(cs_loss, axis=1)]
        for i_a in range(n_agents):
            c_weights = get_network_weights(critics_t[i_a])
            a_weights = get_network_weights(actors_t[i_a])
            for i_w, critic_weight in enumerate(critic_weights):
                critic_weight[i, i_a] = c_weights[i_w] 
            for i_w, actor_weight in enumerate(actor_weights):
                actor_weight[i, i_a] = a_weights[i_w]
        # Test agent
        done = False
        env_state, states = env.reset(seed=seed)
        for i_t in range(step_max):
            actions = [jnp.array(actors_t[i_a](states[i_a])) for i_a in range(n_agents)]
            step_idx = env_state[2]
            env_state,next_states,(rewards,(penalties, is_in_patch)), terminated, truncated, _ = env.step(env_state, *actions)
            agent_states[i, i_t] = env_state[0]
            patch_states[i, i_t] = env_state[1][-1]
            test_penalties[i,step_idx,:] = penalties
            test_is_in_patch[i,step_idx] = is_in_patch
            states = next_states
            done = terminated or truncated
            returns[i,step_idx] = rewards
            if done:
                break
        # Log the important variables to some logger
        (agents_state, patch_state, step_idx) = env_state
        end_energy = agents_state[:, -1]
        log_fun(i, returns[i], end_energy)
        # Compute relevant information
        patch_info = (patch_state[:-1], patch_states)
        env_info = (test_penalties, test_is_in_patch, agent_states, patch_info)
        
    buffer_tuple = zip(*[buffer.get_all() for buffer in buffers])
    buffer_data = [np.concatenate(tuple, axis=0) for tuple in buffer_tuple]
    return returns, ((actors_t, actor_weights), (critics_t, critic_weights)), (actors_loss_stats, critics_loss_stats), env_info, metadata, buffer_data

# TODO: Implement a welfare version of the DDPG algorithm defined above
# Use the Welfare Q-learning algorithm as inspiration
def n_agents_welfare_ddpg(env, num_episodes, tau=0.01, gamma=0.99, batch_size=30, lr_a=2e-4, lr_c=8e-4, seed=0, action_dim=2, state_dim=3, action_max=0.2, hidden_dim=[16,16], log_fun=print_log_ddpg_n_agents):
    # Initialize metadata object for keeping track of (hyper-)parameters and/or additional settings of the environment
    hidden_dims = [str(h_dim) for h_dim in hidden_dim]
    warmup_size = 2*batch_size
    metadata = dict(n_episodes=num_episodes, tau=tau, gamma=gamma, 
                    batch_size=batch_size, lr_actor=lr_a, lr_critic=lr_c, 
                    seed=seed, action_dim=action_dim, state_dim=state_dim,
                    action_max=action_max, hidden_dims=hidden_dims,
                    warmup_size=warmup_size, alg_name="Welfare DDPG", **env.get_params())
    # Initialize neural networks
    n_agents = env.n_agents
    step_max = env.step_max
    actor_dim = action_dim
    
    actors = [Actor(state_dim,actor_dim,action_max,seed+i,hidden_dim=hidden_dim) for i in range(n_agents)]
    actors_t = [Actor(state_dim,actor_dim,action_max,seed+i,hidden_dim=hidden_dim) for i in range(n_agents)]
    critics = [Critic(state_dim+actor_dim,seed+i,out_dim=n_agents,hidden_dim=hidden_dim) for i in range(n_agents)]
    critics_t = [Critic(state_dim+actor_dim,seed+i,out_dim=n_agents,hidden_dim=hidden_dim) for i in range(n_agents)]
    
    optim_actors = [nnx.Optimizer(actors[i], optax.adam(lr_a)) for i in range(n_agents)]
    optim_critics = [nnx.Optimizer(critics[i], optax.adam(lr_c)) for i in range(n_agents)]
    # Add seperate experience replay buffer for each agent 
    buffer_size = num_episodes*env.step_max+warmup_size # Ensure every step is kept in the replay buffer
    buffers = [Buffer(buffer_size, state_dim, actor_dim, reward_dim=n_agents) for i in range(n_agents)]
    # Initialize environment
    key = jax.random.PRNGKey(seed)
    # Keep track of neural network weights (assumes homogeneous networks across agents!)
    critic_weights = [np.empty((num_episodes, n_agents, *shape)) for shape in get_network_shape(critics_t[0])]
    actor_weights = [np.empty((num_episodes, n_agents, *shape)) for shape in get_network_shape(actors_t[0])]
    # Keep track of important loss variables (3 is for [avg,min,max] stats)
    critics_loss_stats = np.empty((num_episodes, 3, n_agents))
    actors_loss_stats = np.empty((num_episodes, 3, n_agents))
    # Keep track of accumulated rewards
    returns = np.empty((num_episodes, step_max, n_agents, 1))
    test_penalties = np.empty((num_episodes, step_max, n_agents, 1))
    # Keep track of patch enter events and states of agents
    test_is_in_patch = np.empty((num_episodes, step_max, n_agents))
    (agents_state, patch_state, step_idx), states = env.reset(seed=seed)
    agent_states = np.empty((num_episodes, step_max, *agents_state.shape))
    patch_states = np.empty((num_episodes, step_max, 1))
    # Warm-up round
    env_state, states = env.reset(seed=seed)
    for s_i in range(warmup_size):
        # Sample action, execute it, and add to buffer
        actions = list(range(n_agents))
        for i_a,actor in enumerate(actors):
            action_key, key = jax.random.split(key)
            actions[i_a] = jnp.array(sample_action(action_key, actor, states[i_a], -action_max, action_max, actor_dim)) 
        env_state, next_states, (rewards, penalties), terminated, truncated, _ = env.step(env_state, *actions)
        (agents_state, patch_state, step_idx) = env_state
        for i_a in range(n_agents):
            buffers[i_a].add(states[i_a], actions[i_a], rewards, next_states[i_a], terminated)
        states = next_states
    
    # Run episodes
    for i in range(num_episodes):
        done = False
        env_state, states = env.reset(seed=seed)
        r_acc = 0
        c = 0
        # Initialize loss temp variables
        cs_loss = np.empty((n_agents,step_max))
        as_loss = np.empty((n_agents,step_max))
        # Train agent
        for s_i in range(step_max):
            # Sample action, execute it, and add to buffer
            actions = list(range(n_agents))
            for i_a,actor in enumerate(actors):
                action_key, key = jax.random.split(key)
                actions[i_a] = jnp.array(sample_action(action_key, actor, states[i_a], -action_max, action_max, actor_dim)) 
            env_state, next_states, (rewards, penalties), terminated, truncated, _ = env.step(env_state, *actions)
            (agents_state, patch_state, step_idx) = env_state
            done = truncated or terminated
            if not terminated:
                for i_a in range(n_agents):
                    # Add states info to buffer
                    buffers[i_a].add(states[i_a], actions[i_a], rewards, next_states[i_a], terminated)
                    # Sample batch from buffer
                    (b_states, b_actions, b_rewards, b_next_states, b_dones), ind = buffers[i_a].sample(batch_size)
                    # Update critic
                    ys = compute_welfare_targets(critics_t[i_a], actors[i_a], b_rewards, b_next_states, b_dones, gamma)
                    c_loss, grads = optimize_welfare_critic(optim_critics[i_a], critics[i_a], b_states, b_actions, ys)
                    # Update policy
                    a_loss, grads = optimize_welfare_actor(optim_actors[i_a], actors[i_a], critics[i_a], b_states)
                    # Update targets (critic and policy)
                    nnx.update(critics_t[i_a], polyak_update(tau, critics_t[i_a], critics[i_a]))
                    nnx.update(actors_t[i_a], polyak_update(tau, actors_t[i_a], actors[i_a]))
                    # Store actor and critic loss 
                    as_loss[i_a,step_idx-1] = a_loss
                    cs_loss[i_a,step_idx-1] = c_loss
            else:
                break
            # Update state of agents
            states = next_states
        # Save training results
        step_idx = env_state[2]
        actors_loss_stats[i,:,:] = [np.mean(as_loss, axis=1), np.min(as_loss, axis=1), np.max(as_loss, axis=1)]
        critics_loss_stats[i,:,:] = [np.mean(cs_loss, axis=1), np.min(cs_loss, axis=1), np.max(cs_loss, axis=1)]
        for i_a in range(n_agents):
            c_weights = get_network_weights(critics_t[i_a])
            a_weights = get_network_weights(actors_t[i_a])
            for i_w, critic_weight in enumerate(critic_weights):
                critic_weight[i, i_a] = c_weights[i_w] 
            for i_w, actor_weight in enumerate(actor_weights):
                actor_weight[i, i_a] = a_weights[i_w]
        # Test agent
        done = False
        env_state, states = env.reset(seed=seed
        c = 0
        for i_t in range(step_max):
            actions = [jnp.array(actors_t[i_a](states[i_a])) for i_a in range(n_agents)]
            step_idx = env_state[2]
            env_state,next_states,(rewards,(penalties, is_in_patch)), terminated, truncated, _ = env.step(env_state, *actions)
            agent_states[i, i_t] = env_state[0]
            patch_states[i, i_t] = env_state[1][-1]
            test_penalties[i,step_idx,:] = penalties
            test_is_in_patch[i,step_idx] = is_in_patch
            states = next_states
            done = terminated or truncated
            returns[i,step_idx] = np.pow(gamma, c)*rewards
            c += 1
            if done:
                break
        # Log the important variables to some logger
        (agents_state, patch_state, step_idx) = env_state
        end_energy = agents_state[:, -1]
        log_fun(i, returns[i], end_energy)
        # Compute relevant information
        patch_info = (patch_state[:-1], patch_states)
        env_info = (test_penalties, test_is_in_patch, agent_states, patch_info)
        
    buffer_tuple = zip(*[buffer.get_all() for buffer in buffers])
    buffer_data = [np.concatenate(tuple, axis=0) for tuple in buffer_tuple]
    return returns, ((actors_t, actor_weights), (critics_t, critic_weights)), (actors_loss_stats, critics_loss_stats), env_info, metadata, buffer_data
        
            