from flax import nnx
import jax.numpy as jnp
import jax
import numpy as np
import optax
import wandb
from environment import *
from save_utils import save_policy, save_policies, load_policies
from functools import partial
import os


## Critic network and it's complementary functions
class Critic(nnx.Module):
    def __init__(self, in_dim, seed, out_dim=1, hidden_dim=[32,32]):
        rngs = nnx.Rngs(seed)
        self.input = nnx.Linear(in_dim, hidden_dim[0], rngs=rngs)
        self.lhs = tuple([nnx.Linear(hidden_dim[i], hidden_dim[i+1], rngs=rngs) for i in range(len(hidden_dim) - 1)])
        self.out = nnx.Linear(hidden_dim[-1], out_dim, rngs=rngs)
    def __call__(self, state, action):
        x = jnp.concatenate([state, action], axis=-1)
        x = nnx.relu(self.input(x))
        for lh in self.lhs:
            x = nnx.relu(lh(x))
        return self.out(x)

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
    def __init__(self, in_dim, out_dim, action_max, seed, hidden_dim=[16,32,16]):
        self.a_max = action_max
        rngs = nnx.Rngs(seed)
        self.input = nnx.Linear(in_dim, hidden_dim[0], rngs=rngs)
        self.lhs = tuple([nnx.Linear(hidden_dim[i], hidden_dim[i+1], rngs=rngs) for i in range(len(hidden_dim) - 1)])
        self.out = nnx.Linear(hidden_dim[-1], out_dim, rngs=rngs)
        self.act_mask = jnp.ones(out_dim)
        if out_dim > 2:
            self.act_mask = self.act_mask.at[2:].set(0)
    def final_activation(self, x):
        return jnp.where(self.act_mask, nnx.tanh(x)*self.a_max, nnx.sigmoid(x))
    def __call__(self, state):
        x = nnx.relu(self.input(state))
        for lh in self.lhs:
            x = nnx.relu(lh(x))
        x = self.out(x)
        return self.final_activation(x)

@nnx.jit
def mean_optimize_actor(optimizer: nnx.Optimizer, actor: nnx.Module, critic: nnx.Module, states: jnp.array):
    loss_fn = lambda actor: -1*critic(states, actor(states)).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(actor)
    optimizer.update(grads)
    return loss, grads

def sample_action(rng, actor, state, action_min, action_max, action_dim, act_noise=0.1):
    mu_action = actor(state)
    eps = jax.random.normal(rng, (action_dim,))*action_max*act_noise
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

def compute_welfare(buffers, lookback):
    rewards = np.zeros(len(buffers))
    for i,buffer in enumerate(buffers):
        size = buffer.get_size()
        rs = buffer.get(slice(size-lookback-1, size))[2]
        rewards[i] = rs.mean()
    NSW = compute_NSW(rewards)
    return NSW

    

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
    def get_size(self):
        return self.size
    def sample(self, batch_size):
        ind = self.rng.integers(0,self.size,size=batch_size)
        return (
            jax.device_put(self.states[ind]),
            jax.device_put(self.actions[ind]),
            jax.device_put(self.rewards[ind]),
            jax.device_put(self.next_states[ind]),
            jax.device_put(self.dones[ind]),
        ), ind

def print_log_ddpg_n_agents(epoch, returns, energies):
    str_out = ""
    str_out += f"Episode {epoch}; "
    str_out += "".join([f"A{i+1} Return: {np.sum(returns[:,i], axis=0):.2f}; " for i in range(returns.shape[1])])
    str_out += "".join([f"A{i+1} Energy: {energy:.2f}; " for i,energy in enumerate(energies)])
    str_out += "                    "
    print(str_out, end="\r")

def create_data_files(path, **info):
    data_path = os.path.abspath(os.path.join(path, "data"))
    make_file = lambda fname, shape: np.memmap(os.path.join(data_path, fname), dtype="float32", mode="w+", shape=shape)
    data = {
        "critics_loss": make_file("critics_loss.dat", (info["num_episodes"], 3, info["n_agents"])),
        "actors_loss": make_file("actors_loss.dat", (info["num_episodes"], 3, info["n_agents"])),
        "returns": make_file("returns.dat", (info["num_episodes"], info["step_max"], info["n_agents"])),
        "penalties": make_file("test_penalties.dat", (info["num_episodes"], info["step_max"], info["n_agents"],1)),
        "is_in_patch": make_file("is_in_patch.dat", (info["num_episodes"], info["step_max"], info["n_agents"])),
        "agent_states": make_file("agent_states.dat", (info["num_episodes"], info["step_max"], *info["agents_state_shape"])),
        "patch_states": make_file("patch_states.dat", (info["num_episodes"], info["step_max"], 1+info["patch_resize"])),
        "actions": make_file("actions.dat", (info["num_episodes"], info["step_max"], info["n_agents"], info["action_dim"])) 
    }
    return data

def n_agents_ddpg(env, num_episodes, tau=0.0025, gamma=0.99, batch_size=240, lr_a=3e-4, lr_c=1e-3, seed=0, action_dim=2, state_dim=9, action_range=[[-4,0],[4,1]], hidden_dim=[256,32], act_noise=0.13, log_fun=print_log_ddpg_n_agents, current_path="", **kwargs):
    # Initialize metadata object for keeping track of (hyper-)parameters and/or additional settings of the environment
    hidden_dims = [str(h_dim) for h_dim in hidden_dim]
    action_range_str = [[str(type[0]), str(type[-1])]  for type in action_range]
    action_range = jnp.array(action_range)
    warmup_size = 5*batch_size
    metadata = dict(n_episodes=num_episodes.item(), tau=tau, gamma=gamma, 
                    batch_size=batch_size, lr_actor=lr_a, lr_critic=lr_c, 
                    action_dim=action_dim, state_dim=state_dim,
                    action_range=action_range_str, hidden_dims=hidden_dims,
                    warmup_size=warmup_size, act_noise=act_noise, alg_name="Normal DDPG", 
                    current_path=current_path, **env.get_params())
    # Initialize neural networks
    n_agents = env.n_agents
    step_max = env.step_max
    patch_resize = env.patch_resize
    actor_dim = action_dim
    action_max = action_range[1][0]
    
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
    (agents_state, patch_state, step_idx), states = env.reset(seed=seed)
    data = create_data_files(current_path, num_episodes=num_episodes, n_agents=n_agents, step_max=step_max, agents_state_shape=agents_state.shape, patch_resize=patch_resize, action_dim=actor_dim)
    # Warm-up round
    print("Filling buffer with warmup samples...", end="\r")
    env_state, states = env.reset(seed=seed)
    for s_i in range(warmup_size):
        # Sample action, execute it, and add to buffer
        actions = list(range(n_agents))
        for i_a,actor in enumerate(actors):
            action_key, key = jax.random.split(key)
            actions[i_a] = jax.random.uniform(action_key, actor_dim, minval=action_range[0], maxval=action_range[1]) 
        env_state, next_states, (rewards, penalties), terminated, truncated, _ = env.step(env_state, *actions)
        (agents_state, patch_state, step_idx) = env_state
        for i_a in range(n_agents):
            buffers[i_a].add(states[i_a], actions[i_a], rewards[i_a], next_states[i_a], terminated)
        states = next_states
    # Run episodes
    print("Training started...                   ", end="\r")
    for i in range(num_episodes):
        done = False
        env_state, states = env.reset(seed=seed+i) # We initialize randomly each episode to allow more exploration
        # Initialize loss temp variables
        cs_loss = np.empty((n_agents,step_max))
        as_loss = np.empty((n_agents,step_max))
        # Train agent
        for s_i in range(step_max):
            # Sample action, execute it, and add to buffer
            actions = list(range(n_agents))
            for i_a,actor in enumerate(actors):
                action_key, key = jax.random.split(key)
                actions[i_a] = np.array(sample_action(action_key, actor, states[i_a], action_range[0], action_range[1], actor_dim, act_noise))
            env_state, next_states, (rewards, penalties), terminated, truncated, _ = env.step(env_state, *actions)
            (agents_state, patch_state, step_idx) = env_state
            done = truncated or terminated
            if terminated:
                break
                
            for i_a in range(n_agents):
                # Add states info to buffer
                buffers[i_a].add(states[i_a], actions[i_a], rewards[i_a], next_states[i_a], terminated)
                # Sample batch from buffer
                (b_states, b_actions, b_rewards, b_next_states, b_dones), ind = buffers[i_a].sample(batch_size)
                # Update critic
                ys = compute_targets(critics_t[i_a], actors_t[i_a], b_rewards, b_next_states, b_dones, gamma)
                c_loss, grads = MSE_optimize_critic(optim_critics[i_a], critics[i_a], b_states, b_actions, ys)
                # Update policy
                a_loss, grads = mean_optimize_actor(optim_actors[i_a], actors[i_a], critics[i_a], b_states)
                # Update targets (critic and policy)
                nnx.update(critics_t[i_a], polyak_update(tau, critics_t[i_a], critics[i_a]))
                nnx.update(actors_t[i_a], polyak_update(tau, actors_t[i_a], actors[i_a]))
                # Store actor and critic loss 
                as_loss[i_a,step_idx-1] = a_loss
                cs_loss[i_a,step_idx-1] = c_loss
            # Update state of agents
            states = next_states
        # Save training results
        step_idx = env_state[2]
        data["actors_loss"][i,:,:] = [np.mean(as_loss, axis=1), np.min(as_loss, axis=1), np.max(as_loss, axis=1)]
        data["critics_loss"][i,:,:] = [np.mean(cs_loss, axis=1), np.min(cs_loss, axis=1), np.max(cs_loss, axis=1)]
        for i_a in range(n_agents):
            c_weights = get_network_weights(critics_t[i_a])
            a_weights = get_network_weights(actors_t[i_a])
            for i_w, critic_weight in enumerate(critic_weights):
                critic_weight[i, i_a] = c_weights[i_w] 
            for i_w, actor_weight in enumerate(actor_weights):
                actor_weight[i, i_a] = a_weights[i_w]
        # Test agent
        done = False
        env_state, states = env.reset(seed=seed+i) # Make sure the reset seed is the same as for training
        for i_t in range(step_max):
            for i_a in range(n_agents):
                data["actions"][i,i_t,i_a,:] = actors_t[i_a](states[i_a])
            step_idx = env_state[2]
            env_state,next_states,(rewards,(penalties, is_in_patch)), terminated, truncated, _ = env.step(env_state, *data["actions"][i,i_t])
            data["agent_states"][i, i_t] = env_state[0]
            data["patch_states"][i, i_t] = env_state[1][-1-patch_resize:]
            data["penalties"][i,step_idx] = penalties
            data["is_in_patch"][i,step_idx] = is_in_patch
            states = next_states
            done = terminated or truncated
            data["returns"][i,step_idx] = rewards
            if done:
                break
        # Log the important variables to some logger
        (agents_state, patch_state, step_idx) = env_state
        end_energy = agents_state[:, 4]
        log_fun(i, data["returns"][i], end_energy)
        # Compute relevant information
        patch_info = (patch_state[:-1], data["patch_states"])
        env_info = (data["penalties"], data["is_in_patch"], data["agent_states"], patch_info)
        # Save all written data to disk indefinitely
        [d.flush() for d in data.values()]
    # Gather buffer data for storing purposes
    buffer_tuple = zip(*[buffer.get_all() for buffer in buffers])
    buffer_data = [np.concatenate(tuple, axis=0) for tuple in buffer_tuple]
    # Save learned actor and critic
    if current_path != "":
        print("\n Saving actor and critic networks of agents...          ")
        save_policies(actors_t, "actors", current_path)
        save_policies(critics_t, "critics", current_path)
        metadata["current_path"] = os.path.abspath(current_path)
    
    return data["returns"], ((actors_t, actor_weights), (critics_t, critic_weights)), (data["actors_loss"], data["critics_loss"]), env_info, metadata, buffer_data