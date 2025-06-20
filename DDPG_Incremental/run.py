from environment import *
from save_utils import load_policy, load_policies, save_policy
from td3 import Actor, n_agents_ddpg
from welfare_functions import *
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from prototype_dashboards import *
import json
import argparse

def create_exp_folder(exp_name, test=False):
    folder_name = "runs"
    if test:
        folder_name= "tests"
    timepoint = datetime.now().strftime("%d-%m-%Y %H%M%S")
    main_folder = os.path.join(folder_name, exp_name)
    path = os.path.join(main_folder, timepoint)
    if not os.path.exists(path):
        os.makedirs(path)
    return os.path.abspath(main_folder), path

def save_metadata(metadata, path):
    with open(os.path.join(path, "metadata.json"), 'w') as f:
        return json.dump(metadata, f, indent=4)

def load_metadata(path):
    with open(os.path.join(path, "metadata.json"), 'r') as f:
        return json.load(f)

def run_ddpg(env, num_episodes, train_fun, path, train_args=dict(), prev_path=None, skip_vid=False):
    # Extract/define initial variables
    episodes = np.arange(1,num_episodes+1)
    action_dim, a_range = env.get_action_space()
    state_dim = env.get_state_space()
    print(f"Action dimension: {action_dim}; State dimension: {state_dim}")
    train_args = {
        "state_dim":state_dim,
        "action_dim":action_dim,
        "action_range":a_range,
        "current_path":path,
        **train_args
    }
    if prev_path is not None:
        metadata = load_metadata(prev_path)
        metadata["previous_path"] = prev_path
        metadata["seed"] = metadata["seed"] + metadata["n_episodes"]
        train_args = metadata
        train_args["current_path"] = path
        
    # Train agent(s)
    train_data, networks, agents_info, metadata, buffer_data = train_fun(env, episodes[-1], **train_args)
    (actor, critic) = networks
    (penalties, is_in_patch, agent_states, patch_info) = agents_info
    # Save all data (as efficiently as possible)
    data = {
        "patch_state": patch_info[0],
        "b_states": buffer_data[0],
        "b_actions": buffer_data[1],
        "b_rewards": buffer_data[2],
        "b_next_states": buffer_data[3]
    }
    np.savez(os.path.join(path, "data"), **data)
    # Save metadata
    save_metadata(metadata, path)
    
    # Plot a few informative plots
    plot_rewards(path, train_data["returns"])
    plot_loss(path, "critic", train_data["critics_loss"])
    plot_loss(path, "actor", train_data["actors_loss"])
    plot_penalty(path, is_in_patch, penalties[:,:,:,0], "action")
    plot_final_welfare(path, train_data["agent_states"], metadata)
    plot_cvals(path, train_data["critics_vals"])
    plot_succes_rate_comm(path, train_data["actions"])
    
    # Draw run of agents over the episodes and save informative plots of final state environment
    plot_final_states_env(path, is_in_patch, patch_info, agent_states[-1], train_data["returns"][-1])
    
    a_shape = (metadata["n_episodes"], metadata["step_max"], metadata["n_agents"], metadata["action_dim"])
    d_path = os.path.abspath(os.path.join(metadata["current_path"], "data"))
    actions = train_data["actions"]
    # Only plot the environment if video toggle is on
    plot_episode_env = lambda episode, path: None
    if not skip_vid:
        plot_episode_env = lambda episode, path: plot_env(path, episode, env.size(), patch_info, agent_states, actions)
    if metadata["n_agents"] == 2:
        episode_data = rq1_data(patch_info, agent_states, actions)
        plot_comm_frequency(path, episode_data[-1])
        episode_results(path, *episode_data, plot_env=plot_episode_env)

def patch_test_saved_policy(env, path, hidden_dim=32):
    state_dim = env.get_state_space()[1]
    action_dim, action_max = env.get_action_space()
    policy = Actor(state_dim, action_dim, action_max[1], 0, hidden_dim=hidden_dim)
    load_policy(policy, path)
    env = RenderOneAgentEnvironment(env)
    state, info = env.reset(seed=0)
    while True:
        state, reward, terminated, truncated, _ = env.step(policy(state))
        if terminated or truncated:
            break
    env.render()

def run_experiment(**kwargs):
    e = kwargs["episodes"]
    s = kwargs["seed"]
    env = NAgentsEnv(**kwargs)
    for i_r in range(kwargs["runs"]):
        main_folder, path = create_exp_folder(kwargs["out"])
        os.mkdir(os.path.join(path, "data"))
        train_args=dict(**kwargs)
        train_args["seed"] = s+i_r*e
        # Single run of DDPG on the environment
        run_ddpg(env, e, n_agents_ddpg, path, train_args, skip_vid=not kwargs["video"])
    # Compute the average return over the amount of runs done
    get_grouped_return(main_folder)

"""
Function used to do inference on trained policy.
Parameters:
- path: Location on device of training session
- num_episodes: How many episodes we want to evaluate the learned actor on

Returns: (states, actions, rewards, metadata)
"""
def run_actor_test(path, num_episodes):
    # Get metadata
    metadata = json.load(open(os.path.join(path, "metadata.json"), "r"))
    # Initialize and load actor model
    hidden_dims = [int(hd) for hd in metadata["hidden_dims"]]
    actor = Actor(metadata["state_dim"], metadata["action_dim"], metadata["v_max"], 0, hidden_dims)
    load_policies([actor], "actors", path)
    # Initialize data variables
    all_states = np.empty((num_episodes, metadata["step_max"]+1, metadata["n_agents"], metadata["state_dim"]))
    all_actions = np.empty((num_episodes, metadata["step_max"], metadata["n_agents"], metadata["action_dim"]))
    returns = np.empty((num_episodes, metadata["step_max"], metadata["n_agents"]))
    # Initialize environment and run episodes
    env = NAgentsEnv(**metadata)
    key = jax.random.key(metadata["seed"]+1)
    for e_i in range(num_episodes):
        print(f"Running episode {e_i}...", end="\r")
        env_state, states = env.reset(key)
        all_states[e_i, 0] = states
        for s_i in range(metadata["step_max"]):
            subkey, key = jax.random.split(key)
            actions = [actor(states[i_a]) for i_a in range(metadata["n_agents"])]
            env_state, next_states, (rewards, penalties), terminated, truncated, _ = env.step(subkey, env_state, *actions)
            all_actions[e_i, s_i] = actions 
            all_states[e_i, s_i+1] = states
            returns[e_i, s_i] = rewards
            states = next_states
    # Return data
    return all_states, all_actions, returns, metadata 

def run_multi_actor_test(path, num_episodes):
    states = []
    actions = []
    returns = []
    runs = [ f.path for f in os.scandir(path) if f.is_dir()]
    for path in runs:
        print(f"Testing {path}...")
        s, a, r, metadata = run_actor_test(path, num_episodes)
        states.append(s)
        actions.append(a)
        returns.append(r)
    states = np.concatenate(states, axis=0)
    actions = np.concatenate(actions, axis=0)
    returns = np.concatenate(returns, axis=0)
    return states, actions, returns, metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=int, help="Number of episodes")
    parser.add_argument("runs", type=int, help="Number of runs")
    parser.add_argument("out", type=str, help="Output folder for results")
    parser.add_argument("-na", "--n-agents", type=int, default=1)
    parser.add_argument("-pw","--p-welfare", type=float, default=0.0, help="Adjusts the proportion of Nash Social Welfare (NSW) used in the reward of agents")
    parser.add_argument("-s", "--seed", type=int, default=0, help="General seed on which we generate our random numbers for the script")
    parser.add_argument("-t", "--tau", type=float, default=0.005, help="Polyak parameter (between 0 and 1) for updating the neural networks")
    parser.add_argument("-g", "--gamma", type=float, default=0.99, help="Discount parameter for Bellman updates of networks")
    parser.add_argument("-b","--batch-size", type=int, default=80, help="Size of the batches used for training updates")
    parser.add_argument("-lra","--lr-a", type=float, default=3e-4, help="Learning rate for the actor network")
    parser.add_argument("-lrc","--lr-c", type=float, default=1e-3, help="Learning rate for the critic network")
    parser.add_argument("-an", "--act-noise", type=float, default=1, help="Adjust the exploration noise added to actions of agents")
    parser.add_argument("--msg-type", nargs="*", type=int, default=[], help="Choose zero or more messages that should be used by the agent: \n 0. Energy \n 1. Position \n 2. Velocity \n 3. Action \n Choosing no message will produce a channel that directly uses the communication value, so the agents learn from communication by themselves.")
    parser.add_argument("--comm-type", type=int, default=0, help="Choose type of communication: \n 0. No communication \n 1. Communication \n 2. Always Communication \n 3. Only Noise")
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, help="Toggle for video generation of episodes")
    parser.add_argument("-pc", "--p-comm", type=float, default=0, help="Communication penalty scalar")
    parser.add_argument("-pt", "--p-att", type=float, default=0, help="Attention penalty scalar")
    parser.add_argument("-pa", "--p-act", type=float, default=0, help="Action penalty scalar")
    parser.add_argument("-ps", "--p-still", type=float, default=0, help="Resting penalty scalar")
    parser.add_argument("-d", "--env-gamma", type=float, default=0.01, help="Decay parameter of patch resource")
    parser.add_argument("-gr", "--eta", type=float, default=0.1, help="Growth parameter of patch resource")
    parser.add_argument("-ba", "--beta", type=float, default=0.1, help="Eating rate of agents")
    parser.add_argument("-si", "--s-init", type=float, default=10, help="Initial amount of resources")
    args = parser.parse_args()
    run_experiment(**vars(args))