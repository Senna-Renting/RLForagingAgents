import wandb
from environment import *
from save_utils import load_policy, save_policy
from td3 import Actor, wandb_train_ddpg, n_agents_train_ddpg
from welfare_functions import *
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from prototype_dashboards import *

def create_exp_folder(exp_name, test=False):
    folder_name = "runs"
    if test:
        folder_name= "tests"
    timepoint = datetime.now().strftime("%d-%m-%Y %H%M%S")
    path = os.path.join(folder_name, exp_name, timepoint)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def ddpg_train_patch_n_agents(env, num_episodes, seed=0, path="", exp_num=1):
    # Extract/define initial variables
    episodes = np.arange(1,num_episodes+1)
    action_dim, a_range = env.get_action_space()
    train_args = {
        "seed":seed,
        "state_dim":env.get_state_space(),
        "action_dim":action_dim,
        "action_max":a_range[1]
    }
    # Train agent(s)
    rewards, networks, (as_loss, cs_loss), agents_info, reset_key = n_agents_train_ddpg(env, episodes[-1], **train_args)
    ((actors, a_weights), (critics, c_weights)) = networks
    (penalties, is_in_patch, agent_states, patch_info) = agents_info

    # Save all data (as efficiently as possible)
    data = {
        "rewards": rewards,
        "penalties": penalties,
        "is_in_patch": is_in_patch,
        "patch_state": patch_info[0],
        "patch_resource": patch_info[1],
        "agent_states": agent_states
    }
    np.savez(os.path.join(path, "data"), **data)
    
    
    # Plot a few informative plots
    plot_rewards(path, rewards)
    plot_loss(path, "critic", cs_loss)
    plot_loss(path, "actor", as_loss)
    plot_penalty(path, is_in_patch, penalties[:,:,:,0], "action")
    if penalties.shape[3] == 2:
        plot_penalty(path, is_in_patch, penalties[:,:,:,1], "communication")
    
    # Draw run of agents over the episodes and save informative plots of final state environment
    plot_final_states_env(path, is_in_patch, patch_info, agent_states[-1], rewards[-1])
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

"""
For this experiment we test the single-agent one-patch environment
Later I will extend this to multiple runs and use those to generate statistics for significance testing
"""
def experiment1(num_episodes, num_runs, test=False):
    for i in range(num_runs):
        path = create_exp_folder("Experiment1", test=test)
        print(f"Run {i+1} has been started")
        env = NAgentsEnv(n_agents=1, seed=i)
        ddpg_train_patch_n_agents(env, num_episodes, seed=i, path=path)

"""
For this experiment we test the two-agent one-patch environment
The agents don't observe each other, and do not communicate.
Later I will extend this to multiple runs and use those to generate statistics for significance testing
"""
def experiment2(num_episodes, num_runs, test=False):
    for i in range(num_runs):
        path = create_exp_folder("Experiment2", test=test)
        print(f"Run {i+1} has been started")
        env = NAgentsEnv(n_agents=2, seed=i)
        ddpg_train_patch_n_agents(env, num_episodes, seed=i, path=path)

"""
For this experiment we test the two-agent one-patch environment
The agents observe each other, but do not communicate.
Later I will extend this to multiple runs and use those to generate statistics for significance testing
"""
def experiment3(num_episodes, num_runs, test=False):
    for i in range(num_runs):
        path = create_exp_folder("Experiment3", test=test)
        print(f"Run {i+1} has been started")
        env = NAgentsEnv(n_agents=2, obs_others=True, seed=i)
        ddpg_train_patch_n_agents(env, num_episodes, seed=i, path=path)

"""
For this experiment we test the two-agent one-patch environment
The agents observe each other, and communicate via a social welfare function provided as a reward signal.
Later I will extend this to multiple runs and use those to generate statistics for significance testing
"""
def experiment4(num_episodes, num_runs, test=False):
    for i in range(num_runs):
        path = create_exp_folder("Experiment4", test=test)
        env = NAgentsEnv(n_agents=2, obs_others=True, seed=i, sw_fun=nash_sw, p_welfare=0.2)
        ddpg_train_patch_n_agents(env, num_episodes, seed=i, path=path)

"""
For this experiment we test the single-agent one-patch environment
The agents observe each other, and communicate via by providing a message in addition to an action for their policy.
We will use the messages as state inputs, to train the critic on
Later I will extend this to multiple runs and use those to generate statistics for significance testing
"""
def experiment5(num_episodes, num_runs, test=False):
    for i in range(num_runs):
        path = create_exp_folder("Experiment5", test=test)
        env = NAgentsEnv(n_agents=2, obs_others=False, seed=i, comm_dim=1, sw_fun=nash_sw, p_welfare=0.7)
        ddpg_train_patch_n_agents(env, num_episodes, seed=i, path=path)


if __name__ == "__main__":
    # Experiments can be run below
    experiment4(2,1, test=True)
    
    # num_episodes = 5
    # num_runs = 5
    
    # Uncomment the environment needed below
    #env = NAgentsEnv(patch_radius=0.5, step_max=400, alpha=0.025, beta=0.5, e_init=1, n_agents=2, obs_others=False, seed=2)
    
    # Uncomment the method needed below
    #ddpg_train_patch_n_agents(env, num_episodes, path=path)
    #wandb_ddpg_train_patch(env, num_episodes, num_runs=num_runs, hidden_dim=256, batch_size=100, warmup_steps=200)
    
    # Fill in the path of the policy and uncomment the method below it
    #path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policies", "e4nii8kg", "efficient-sweep-4")
    #patch_test_saved_policy(env, path, hidden_dim=256)