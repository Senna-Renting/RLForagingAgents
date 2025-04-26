import wandb
from environment import *
from save_utils import load_policy, save_policy
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
    ((actors, a_weights), (critics, c_weights)) = networks
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
    np.savez(os.path.join(path, "actor_weights"), *a_weights)
    np.savez(os.path.join(path, "critic_weights"), *c_weights)
    # Save metadata
    save_metadata(metadata, path)
    
    # Plot a few informative plots
    plot_rewards(path, train_data["returns"])
    plot_loss(path, "critic", train_data["critics_loss"])
    plot_loss(path, "actor", train_data["actors_loss"])
    plot_penalty(path, is_in_patch, penalties[:,:,:,0], "action")
    plot_final_welfare(path, agent_states)
    
    # Draw run of agents over the episodes and save informative plots of final state environment
    plot_final_states_env(path, is_in_patch, patch_info, agent_states[-1], train_data["returns"][-1])
    
    a_shape = (metadata["n_episodes"], metadata["step_max"], metadata["n_agents"], metadata["action_dim"])
    d_path = os.path.abspath(os.path.join(metadata["current_path"], "data"))
    actions = np.memmap(os.path.join(d_path, "actions.dat"), mode="r", dtype="float32", shape=a_shape)
    # Only plot the environment if video toggle is on
    plot_episode_env = lambda episode, path: None
    if not skip_vid:
        plot_episode_env = lambda episode, path: plot_env(path, episode, env.size(), patch_info, agent_states, actions)
    if metadata["n_agents"] == 2:
        episode_results(path, *rq1_data(patch_info, agent_states, actions), plot_env=plot_episode_env)

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
    s = kwargs["seed"]
    e = kwargs["episodes"]
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=int, help="Number of episodes")
    parser.add_argument("runs", type=int, help="Number of runs")
    parser.add_argument("out", type=str, help="Output folder for results")
    parser.add_argument("-na", "--n-agents", type=int, default=1)
    parser.add_argument("--obs-others", action=argparse.BooleanOptionalAction, default=False, help="Toggle to allow agents to observe each other")
    parser.add_argument("-ipo", "--in-patch-only", action=argparse.BooleanOptionalAction, default=True, help="Toggle for agents to only observe the state of the patch when inside")
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
    parser.add_argument("--rof", type=int, default=0, help="Size of ring of fire around patch")
    parser.add_argument("--patch-resize", action=argparse.BooleanOptionalAction, default=False, help="Allow the patch to resize based on the amount of resources present")
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, help="Toggle for video generation of episodes")
    args = parser.parse_args()
    run_experiment(**vars(args))