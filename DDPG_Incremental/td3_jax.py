from env_jax import *
from td3 import *
import os
from run import create_exp_folder

def create_data_files(**metadata):
    data_path = os.path.abspath(os.path.join(metadata["current_path"], "data"))
    os.mkdir(data_path)
    make_file = lambda fname, shape: np.memmap(os.path.join(data_path, fname), dtype="float32", mode="w+", shape=shape)
    n_ep = metadata["num_episodes"]
    s_max = metadata["step_max"]
    b_size = metadata["batch_size"]
    steps = n_ep*s_max
    n_ag = metadata["n_agents"]
    data = {
        "critics_vals": make_file("critics_vals.dat", (s_max*b_size,n_ag, 2)),
        "critics_loss": make_file("critics_loss.dat", (steps, n_ag)),
        "actors_loss": make_file("actors_loss.dat", (steps, n_ag)),
        "returns": make_file("returns.dat", (n_ep, s_max, n_ag)),
        "agent_states": make_file("agent_states.dat", (n_ep, s_max, metadata["state_dim"])),
        "patch_states": make_file("patch_states.dat", (n_ep, s_max, 1)),
        "actions": make_file("actions.dat", (n_ep, s_max, n_ag, metadata["action_dim"])) 
    }
    return data

# TODO: re-implement the ddpg algorithm from the td3.py file to work on the EnvState implementation
def jax_n_agents_ddpg(train_params: TrainParameters, env_params: EnvParameters):
    # Extract useful variables from parameters
    n_agents = env_params.n_agents
    actor_dim = 2*(1+(env_params.comm_type>0))
    current_path, hidden_dims, action_range, batch_size, n_episodes, seed, step_max, tau, gamma, act_noise, lr_a, lr_c = train_params
    minvals = jnp.array([action_range[0] for i_a in range(n_agents)])
    maxvals = jnp.array([action_range[1] for i_a in range(n_agents)])
    print("Minval, Maxval: ", minvals.shape, maxvals.shape)
    action_max = action_range[1][0]
    # Initialize environment
    key = jax.random.PRNGKey(seed)
    state, obs = env_reset(key, env_params)
    state_dim = obs.shape[1]
    # Initialize metadata object for keeping track of (hyper-)parameters and/or additional settings of the environment
    warmup_size = 5*batch_size
    train_dict = train_params._asdict().copy()
    train_dict["hidden_dims"] = [str(h_dim) for h_dim in hidden_dims]
    train_dict["action_range"] = [[str(min), str(max)]  for min,_,max,_ in action_range]
    metadata = dict(action_dim=actor_dim, state_dim=state_dim, **train_dict, **env_params._asdict())
    print(metadata)
    # Create memory mapped files
    data = create_data_files(**metadata)
    # Initialize normal and target networks
    actors = [Actor(state_dim,actor_dim,action_max,seed+i_a,hidden_dim=hidden_dims) for i_a in range(n_agents)]
    actors_t = [Actor(state_dim,actor_dim,action_max,seed+i_a,hidden_dim=hidden_dims) for i_a in range(n_agents)]
    critics = [Critic(state_dim+actor_dim,seed+i_a,out_dim=1,hidden_dim=hidden_dims) for i_a in range(n_agents)]
    critics_t = [Critic(state_dim+actor_dim,seed+i_a,out_dim=1,hidden_dim=hidden_dims) for i_a in range(n_agents)]
    # Initialize optimizers for actor and critic
    optim_actors = [nnx.Optimizer(actors[i_a], optax.adam(train_params.lr_a)) for i_a in range(n_agents)]
    optim_critics = [nnx.Optimizer(critics[i_a], optax.adam(train_params.lr_c)) for i_a in range(n_agents)]
    # Add seperate experience replay buffer for each agent 
    buffer_size = n_episodes*step_max+warmup_size # Ensure every step is kept in the replay buffer
    buffers = [Buffer(buffer_size, state_dim, actor_dim, seed=seed+i_a) for i_a in range(n_agents)]
    # Warm-up round
    print("Filling buffer with warmup samples...", end="\r")
    for s_i in range(warmup_size):
        # Sample action, execute it, and add to buffer
        step_key, action_key, key = jax.random.split(key,3)
        actions = jax.random.uniform(action_key, (n_agents,actor_dim), minval=minvals, maxval=maxvals) 
        state, next_obs, rewards, done = env_step(state, actions, step_key, env_params)
        for i_a in range(n_agents):
            buffers[i_a].add(obs.at[i_a].get(), actions.at[i_a].get(), rewards.at[i_a].get(), next_obs.at[i_a].get(), done)
        obs = next_obs
    # Run episodes
    print("Training started...                   ", end="\r")
    for i in range(n_episodes):
        state, obs = env_reset(key, env_params) # We initialize randomly each episode to allow more exploration
        # Initialize loss temp variables
        cs_loss = jnp.empty((step_max,n_agents))
        c_vals = jnp.empty((step_max*batch_size,n_agents,2)) # 2 is for learned and target critic respectively 
        as_loss = jnp.empty((step_max,n_agents))
        # Train agent
        for s_i in range(step_max):
            # Sample action and execute it
            key, step_key, *keys = jax.random.split(key,4)
            actions = jnp.array([sample_action(keys[i_a], actors[i_a], obs[i_a], minvals[i_a], 
                                               maxvals[i_a], act_noise) for i_a in range(n_agents)])
            state, next_obs, rewards, done = env_step(state, actions, step_key, env_params)
            for i_a in range(n_agents):
                # Add states info to buffer
                buffers[i_a].add(obs.at[i_a].get(), actions.at[i_a].get(), rewards.at[i_a].get(), next_obs.at[i_a].get(), done)
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
                # Compute Critic-Critic target difference
                c_start = batch_size*(s_i-1)
                c_end = c_start + batch_size
                c_vals = c_vals.at[c_start:c_end,i_a,0].set(critics[i_a](b_states, b_actions).flatten())
                c_vals = c_vals.at[c_start:c_end,i_a,1].set(critics_t[i_a](b_states, b_actions).flatten())
                # Store actor and critic loss 
                as_loss = as_loss.at[s_i-1,i_a].set(a_loss)
                cs_loss = cs_loss.at[s_i-1,i_a].set(c_loss)
            # Update observations of agents
            obs = next_obs
        # Save training results
        data["actors_loss"][i*step_max:(i+1)*step_max,:] = jax.device_put(as_loss)
        data["critics_loss"][i*step_max:(i+1)*step_max,:] = jax.device_put(cs_loss)
        # Test agent
        state, obs = env_reset(key, env_params)
        for i_t in range(step_max):
            for i_a in range(n_agents):
                data["actions"][i,i_t,i_a,:] = np.asarray(actors_t[i_a](obs[i_a]))
            state,next_obs,rewards, done = env_step(state, jax.device_put(data["actions"][i,i_t])
            data["agent_states"][i, i_t] = np.asarray(get_agent_states(state))
            data["patch_states"][i, i_t] = np.asarray(get_patch_state(state))
            data["returns"][i, i_t] = np.asarray(rewards)
            obs = next_obs
        # Save all written data to disk indefinitely
        [d.flush() for k,d in data.items() if k != "critics_vals"]
        # Log the important variables to some logger
        end_energy = [state.agent_states[i_a].energy for i_a in range(n_agents)]
        log_fun(i, data["returns"][i], end_energy)
        # Compute relevant information
        patch_info = (state.patch_state.resource, data["patch_states"])
        env_info = (data["agent_states"], patch_info)
    # Store the critic values on disk
    data["critics_vals"][:,:,:] = c_vals
    data["critics_vals"].flush()
    # Save learned actor and critic
    print("\n Saving actor and critic networks of agents...          ")
    save_policies(actors_t, "actors", current_path)
    save_policies(critics_t, "critics", current_path)
    metadata["current_path"] = os.path.abspath(current_path)
    return data, metadata

def run_jax_ddpg(train_fun, path):
    env_params =  EnvParameters(env_size = 50,
                                p_welfare = 0.9,
                                dt = 0.1,
                                p_att = 0.02,
                                p_comm = 0.1,
                                msg_noise = 0.1,
                                comm_type = 1,
                                growth = 0.1,
                                decay = 0.01,
                                patch_resource_init = 10,
                                patch_radius = 10,
                                p_still = 0.02,
                                p_act = 0.2,
                                n_agents = 2,
                                v_max = 4,
                                e_max = 10 + 5,
                                eat_rate = 0.1,
                                e_init = 5,
                                damping = 0.3)
    train_params = TrainParameters(current_path=path, 
                                   hidden_dims=(256,256), 
                                   action_range=((-4,-4,0,0),(4,4,1,1)), 
                                   batch_size=64, 
                                   num_episodes=1, 
                                   seed=0, 
                                   step_max=600,
                                   tau=0.005, 
                                   gamma=0.99, 
                                   act_noise=0.1, 
                                   lr_a=1e-3, 
                                   lr_c=3e-4)
    train_fun(train_params, env_params)

if __name__ == "__main__":
    main_path, path = create_exp_folder("test_jax_ddpg")
    run_jax_ddpg(jax_n_agents_ddpg, path)
