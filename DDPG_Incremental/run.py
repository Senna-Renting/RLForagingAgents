import wandb
from environment import *
from save_utils import load_policy, save_policy
from td3 import Actor, wandb_train_ddpg, n_agents_ddpg, n_agents_td3
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
    path = os.path.join(folder_name, exp_name, timepoint)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

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
    train_args = {
        "state_dim":env.get_state_space(),
        "action_dim":action_dim,
        "action_max":a_range[1],
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
    rewards, networks, (as_loss, cs_loss), agents_info, metadata, buffer_data = train_fun(env, episodes[-1], **train_args)
    ((actors, a_weights), (critics, c_weights)) = networks
    (penalties, is_in_patch, agent_states, patch_info) = agents_info
    # Save all data (as efficiently as possible)
    data = {
        "rewards": rewards,
        "penalties": penalties,
        "is_in_patch": is_in_patch,
        "patch_state": patch_info[0],
        "patch_resource": patch_info[1],
        "agent_states": agent_states,
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
    plot_rewards(path, rewards)
    plot_loss(path, "critic", cs_loss)
    plot_loss(path, "actor", as_loss)
    plot_penalty(path, is_in_patch, penalties[:,:,:,0], "action")
    if penalties.shape[3] == 2:
        plot_penalty(path, is_in_patch, penalties[:,:,:,1], "communication")
    
    # Draw run of agents over the episodes and save informative plots of final state environment
    plot_final_states_env(path, is_in_patch, patch_info, agent_states[-1], rewards[-1])
    
    # Toggle for generating video of the training over episodes
    if not skip_vid:
        plot_env(path, env.size(), patch_info, agent_states)
    
    

def wandb_ddpg_train_patch(env, num_episodes, num_runs=5, hidden_dim=32, batch_size=100, warmup_steps=200):
    episodes = list(range(1,num_episodes+1))
    action_dim, a_range = env.get_action_space()
    wandb.login()
    sweep_config = {
        'method':'bayes',
        'metric':{
            'name':'Return',
            'goal':'maximize'
        },
        'parameters':{
            'lr_c':{
                'distribution':'uniform',
                'min': 5e-4,
                'max': 3e-3
            },
            'lr_a':{
                'distribution':'uniform',
                'min': 5e-5,
                'max': 5e-4
            },
            'tau':{
                'distribution':'uniform',
                'min':0,
                'max':0.3
            },
            'action_dim':{
                'value':action_dim
            },
            'state_dim':{
                'value':env.get_state_space()[1]
            },
            'hidden_dim':{
                'value':hidden_dim
            },
            'batch_size':{
                'value':batch_size
            },
            'num_episodes':{
                'value':num_episodes
            },
            'warmup_steps':{
                'value':warmup_steps
            },
            'seed':{
                'value':0
            },
            'reset_seed':{
                'value':0
            },
            'action_max':{
                'value':a_range[1]
            }
        }
    }
    sweep_id = wandb.sweep(sweep_config, project="OneAgentPatchDDPG")
    train_fun = wandb_train_ddpg(env)
    wandb.agent(sweep_id, train_fun, count=num_runs)

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
    print(kwargs)
    s = kwargs["seed"]
    e = kwargs["episodes"]
    env = NAgentsEnv(n_agents=kwargs["n_agents"], obs_others=kwargs["obs"], p_welfare=kwargs["pw"], obs_range=kwargs["obsrange"], in_patch_only=kwargs["ipo"])
    for i_r in range(kwargs["runs"]):
        path = create_exp_folder(kwargs["out"])
        os.mkdir(os.path.join(path, "data"))
        train_args=dict(seed=s+i_r*e, tau=kwargs["tau"], gamma=kwargs["gamma"], batch_size=kwargs["bsize"], lr_c=kwargs["lrc"], lr_a=kwargs["lra"], act_noise=kwargs["act_noise"])
        run_ddpg(env, e, n_agents_ddpg, path, train_args, skip_vid=not kwargs["video"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=int, help="Number of episodes")
    parser.add_argument("runs", type=int, help="Number of runs")
    parser.add_argument("out", type=str, help="Output folder for results")
    parser.add_argument("-na", "--n-agents", type=int, default=1)
    parser.add_argument("--obs", action=argparse.BooleanOptionalAction, default=False, help="Toggle to allow agents to observe each other")
    parser.add_argument("--ipo", action=argparse.BooleanOptionalAction, default=True, help="Toggle for agents to only observe the state of the patch when inside")
    parser.add_argument("-or", "--obsrange", type=float, default=80, help="Adjusts the distance below which agents observe each other")
    parser.add_argument("--pw", type=float, default=0.0, help="Adjusts the proportion of Nash Social Welfare (NSW) used in the reward of agents")
    parser.add_argument("-s", "--seed", type=int, default=0, help="General seed on which we generate our random numbers for the script")
    parser.add_argument("-t", "--tau", type=float, default=0.01, help="Polyak parameter (between 0 and 1) for updating the neural networks")
    parser.add_argument("-g", "--gamma", type=float, default=0.99, help="Discount parameter for Bellman updates of networks")
    parser.add_argument("--bsize", type=int, default=240, help="Size of the batches used for training updates")
    parser.add_argument("--lra", type=float, default=3e-4, help="Learning rate for the actor network")
    parser.add_argument("--lrc", type=float, default=1e-3, help="Learning rate for the critic network")
    parser.add_argument("-an", "--act-noise", type=float, default=0.2, help="Adjust the exploration noise added to actions of agents")
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, help="Toggle for video generation of episodes")
    args = parser.parse_args()
    run_experiment(**vars(args))
    # Experiments can be run below
    # experiment4(100,1)