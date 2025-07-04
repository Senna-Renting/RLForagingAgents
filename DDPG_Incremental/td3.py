from flax import nnx
import jax.numpy as jnp
import jax
import numpy as np
import optax
from environment import *
from save_utils import save_policies
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
    loss_fn = lambda critic: ((critic(states, actions) - jax.lax.stop_gradient(ys)) ** 2).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(critic)
    optimizer.update(grads)
    return loss, grads

@nnx.jit
def compute_targets(critic_t: nnx.Module, actor_t: nnx.Module, rs: jnp.array, next_states: jnp.array, done: jnp.array, gamma: float):
    q_target = critic_t(next_states, actor_t(next_states))
    return rs + gamma*q_target*(1-done)

## Actor network and it's complementary functions
class Actor(nnx.Module):
    def __init__(self, in_dim, out_dim, action_max, seed, hidden_dim=[16,32,16]):
        rngs = nnx.Rngs(seed)
        self.input = nnx.Linear(in_dim, hidden_dim[0], rngs=rngs)
        self.lhs = tuple([nnx.Linear(hidden_dim[i], hidden_dim[i+1], rngs=rngs) for i in range(len(hidden_dim) - 1)])
        self.out = nnx.Linear(hidden_dim[-1], out_dim, rngs=rngs)
        act_mask = jnp.ones(out_dim)
        if out_dim > 2:
            act_mask = act_mask.at[2:].set(0)
        self.act_mask = nnx.Variable(act_mask)
        self.a_max = nnx.Variable(action_max)
    def final_activation(self, x):
        return jnp.where(self.act_mask.value, nnx.tanh(x)*self.a_max.value, nnx.sigmoid(x))
    def __call__(self, state):
        x = nnx.relu(self.input(state))
        for lh in self.lhs:
            x = nnx.relu(lh(x))
        x = self.out(x)
        return self.final_activation(x)

@nnx.jit
def mean_optimize_actor(optimizer: nnx.Optimizer, actor: nnx.Module, critic: nnx.Module, states: jnp.array):
    def loss_fn(actor, critic, states): 
        actions = actor(states)
        q_vals = critic(states, actions)
        return -jnp.mean(q_vals)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=0)(actor, critic, states)
    optimizer.update(grads)
    return loss, grads

def sample_action(rng, actor, state, action_min, action_max, act_noise=0.1):
    mu_action = actor(state)
    eps = jax.random.normal(rng, mu_action.shape)*action_max*act_noise
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
    def add(self, state, action, reward, next_state, done, size=1):
        self.states[self.ptr:self.ptr+size, :] = state
        self.actions[self.ptr:self.ptr+size, :] = action
        self.rewards[self.ptr:self.ptr+size, :] = reward
        self.next_states[self.ptr:self.ptr+size, :] = next_state
        self.dones[self.ptr:self.ptr+size, :] = done
        self.size = min(self.size + size, self.max_size)
        self.ptr = (self.ptr + size) % self.max_size
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
            self.states[np.newaxis, :self.size],
            self.actions[np.newaxis, :self.size],
            self.rewards[np.newaxis, :self.size],
            self.next_states[np.newaxis, :self.size],
            self.dones[np.newaxis, :self.size],
        )
    def get_size(self):
        return self.size
    def sample(self, batch_size, key):
        ind = np.asarray(jax.random.randint(key, batch_size,minval=0,maxval=self.size))
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
    make_file = lambda fname, shape: np.lib.format.open_memmap(os.path.join(data_path, fname), mode="w+", dtype="float32", shape=shape)
    n_ep = info["num_episodes"]
    s_max = info["step_max"]
    b_size = info["batch_size"]
    steps = n_ep*s_max
    n_ag = info["n_agents"]
    update_every = info["update_every"]
    data = {
        "critics_vals": make_file("critics_vals.npy", (s_max*b_size//update_every, 2)),
        "critics_loss": make_file("critics_loss.npy", (steps//update_every,)),
        "actors_loss": make_file("actors_loss.npy", (steps//update_every,)),
        "returns": make_file("returns.npy", (n_ep, s_max, n_ag)),
        "penalties": make_file("test_penalties.npy", (n_ep, s_max, n_ag, 1)),
        "is_in_patch": make_file("is_in_patch.npy", (n_ep, s_max, n_ag)),
        "agent_states": make_file("agent_states.npy", (n_ep, s_max, *info["agents_state_shape"])),
        "patch_states": make_file("patch_states.npy", (n_ep, s_max, 1)),
        "actions": make_file("actions.npy", (n_ep, s_max, n_ag, info["action_dim"])) 
    }
    return data

def n_agents_ddpg(env, num_episodes, tau=0.0025, gamma=0.99, batch_size=240, lr_a=3e-4, lr_c=1e-3, seed=0, action_dim=2, state_dim=9, action_range=[[-4,0],[4,1]], hidden_dim=[400,300], act_noise=0.13, log_fun=print_log_ddpg_n_agents, current_path="", update_every=1, **kwargs):
    # Initialize metadata object for keeping track of (hyper-)parameters and/or additional settings of the environment
    n_agents = env.n_agents
    step_max = env.step_max
    patch_resize = env.patch_resize
    warmup_size = 10*step_max
    metadata = dict(update_every=update_every, n_episodes=num_episodes.item(), tau=tau, 
                    gamma=gamma, batch_size=batch_size, lr_actor=lr_a, lr_critic=lr_c, 
                    action_dim=action_dim, state_dim=state_dim,
                    action_range=action_range, hidden_dims=hidden_dim,
                    warmup_size=warmup_size, act_noise=act_noise, alg_name="Normal DDPG", 
                    current_path=current_path, seed=seed, **env.get_params())
    action_range = jnp.array(action_range)
    # Initialize neural networks
    actor_dim = action_dim
    action_max = action_range[1][0]
    
    actor = Actor(state_dim,actor_dim,action_max,seed,hidden_dim=hidden_dim)
    actor_t = Actor(state_dim,actor_dim,action_max,seed,hidden_dim=hidden_dim)
    critic = Critic(state_dim+actor_dim,seed,out_dim=1,hidden_dim=hidden_dim)
    critic_t = Critic(state_dim+actor_dim,seed,out_dim=1,hidden_dim=hidden_dim)

    optim_actor = nnx.Optimizer(actor, optax.adam(lr_a))
    optim_critic = nnx.Optimizer(critic, optax.adam(lr_c))
 
    buffer_size = num_episodes*env.step_max*n_agents+warmup_size # Ensure every step is kept in the replay buffer
    buffer = Buffer(buffer_size, state_dim, actor_dim)
    # Initialize environment
    key = jax.random.PRNGKey(seed)
    # Keep track of important variables
    (agents_state, patch_state, step_idx), states = env.reset(key)
    data = create_data_files(current_path, num_episodes=num_episodes, n_agents=n_agents, step_max=step_max, agents_state_shape=agents_state.shape, patch_resize=patch_resize, action_dim=actor_dim, batch_size=batch_size, update_every=update_every)
    # Warm-up round
    print("Filling buffer with warmup samples...", end="\r")
    for s_i in range(warmup_size):
        if s_i % step_max == 0:
            subkey, key = jax.random.split(key)
            env_state, states = env.reset(subkey)
        # Sample action, execute it, and add to buffer
        actions = list(range(n_agents))
        for i_a in range(n_agents):
            step_key, action_key, key = jax.random.split(key,3)
            actions[i_a] = jax.random.uniform(action_key, actor_dim, minval=action_range[0], maxval=action_range[1]) 
        env_state, next_states, (rewards, penalties), terminated, truncated, _ = env.step(step_key, env_state, *actions)
        for i_a in range(n_agents):
            buffer.add(states[i_a], actions[i_a], rewards[i_a], next_states[i_a], terminated)
        states = next_states
    
    # Run episodes
    print("Training started...                   ", end="\r")
    for i in range(num_episodes):
        done = False
        subkey, key = jax.random.split(key)
        env_state, states = env.reset(subkey) # We initialize randomly each episode to allow more exploration
        # Initialize loss temp variables
        cs_loss = np.empty((step_max//update_every))
        c_vals = np.empty((step_max//update_every*batch_size,2)) # 2 is for learned and target critic respectively 
        as_loss = np.empty((step_max//update_every))
        # Train agent
        for s_i in range(step_max):
            # Sample action and execute it
            actions = list(range(n_agents))
            for i_a in range(n_agents):
                step_key,action_key, key = jax.random.split(key,3)
                actions[i_a] = np.array(sample_action(action_key, actor, states[i_a], action_range[0], action_range[1], act_noise))
            env_state, next_states, (rewards, penalties), terminated, truncated, _ = env.step(step_key,env_state, *actions)
            (agents_state, patch_state, step_idx) = env_state
            done = truncated or terminated
            if terminated:
                break
            for i_a in range(n_agents):
                # Add states info to buffer
                buffer.add(states[i_a], actions[i_a], rewards[i_a], next_states[i_a], terminated)
            if s_i % update_every == 0: 
                # Sample batch from buffer
                subkey, key = jax.random.split(key)
                (b_states, b_actions, b_rewards, b_next_states, b_dones), ind = buffer.sample(batch_size, subkey)
                # Update critic
                ys = compute_targets(critic_t, actor_t, b_rewards, b_next_states, b_dones, gamma)
                c_loss, grads = MSE_optimize_critic(optim_critic, critic, b_states, b_actions, ys)
                # Update policy
                a_loss, grads = mean_optimize_actor(optim_actor, actor, critic, b_states)
                # Update targets (critic and policy)
                nnx.update(critic_t, polyak_update(tau, critic_t, critic))
                nnx.update(actor_t, polyak_update(tau, actor_t, actor))
                # Compute Critic-Critic target difference
                if i == num_episodes - 1:
                    c_start = batch_size*(s_i//update_every)
                    c_end = c_start + batch_size
                    c_vals[c_start:c_end,0] = critic(b_states, b_actions).flatten()
                    c_vals[c_start:c_end,1] = critic_t(b_states, b_actions).flatten()
                # Store actor and critic loss 
                as_loss[s_i//update_every] = a_loss
                cs_loss[s_i//update_every] = c_loss
            # Update state of agents
            states = next_states
        # Save training results
        step_idx = env_state[2]
        l_start = i*(step_max//update_every)
        l_end = l_start + step_max//update_every
        data["actors_loss"][l_start:l_end] = as_loss
        data["critics_loss"][l_start:l_end] = cs_loss
        
        # Test agent
        done = False
        subkey, key = jax.random.split(key)
        env_state, states = env.reset(subkey)
        for i_t in range(step_max):
            for i_a in range(n_agents):
                data["actions"][i,i_t,i_a,:] = actor_t(states[i_a])
            step_idx = env_state[2]
            subkey, key = jax.random.split(key)
            env_state,next_states,(rewards,(penalties, is_in_patch)), terminated, truncated, _ = env.step(subkey,env_state, *data["actions"][i,i_t])
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
    # Store the critic values on disk
    data["critics_vals"][:,:] = c_vals
    # Gather buffer data for storing purposes
    buffer_data = buffer.get_all()
    # Save learned actor and critic
    if current_path != "":
        print("\n Saving actor and critic networks of agents...          ")
        save_policies([actor_t], "actors", current_path)
        save_policies([critic_t], "critics", current_path)
        metadata["current_path"] = os.path.abspath(current_path)
    return data, (actor_t, critic_t), env_info, metadata, buffer_data
    
