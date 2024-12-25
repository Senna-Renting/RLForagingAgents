import wandb
from environment import *
from save_utils import load_policy, save_policy
from td3 import Actor, wandb_train_ddpg, train_ddpg, n_agents_train_ddpg
from welfare_functions import *
import os
from datetime import datetime
import matplotlib.pyplot as plt

def create_exp_folder(exp_name):
    timepoint = datetime.now().strftime("%d-%m-%Y %H%M%S")
    path = os.path.join("runs", exp_name, timepoint)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def plot_run_info(path, rewards, social_welfare=None, sw_fun=lambda x:0):
    plt.figure()
    plt.title("Return over episodes for each agent")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    [plt.plot(rewards[:,i], label=f"Agent {i+1}") for i in range(rewards.shape[1])]
    plt.legend()
    plt.savefig(os.path.join(path, "agent_episodes_return.png"))
    if sw_fun.__name__ != "<lambda>":
        plt.figure()
        plt.title("Return over episodes for each agent")
        plt.xlabel("Episode")
        plt.ylabel(f"Social welfare ({sw_fun.__name__})")
        plt.plot(social_welfare)
        plt.savefig(os.path.join(path, "episodes_welfare.png"))

def ddpg_train_patch(env, num_episodes):
    path = create_exp_folder("Experiment1")
    episodes = list(range(1,num_episodes+1))
    action_dim, a_range = env.get_action_space()
    # Train agent
    rewards, actor, critic, reset_key = train_ddpg(env, episodes[-1], lr_c=1e-3, lr_a=3e-4, tau=0.01, action_dim=action_dim, state_dim=env.get_state_space()[1], action_max=a_range[1], hidden_dim=64, batch_size=100, seed=0, reset_seed=0)

    # Plot and save rewards figure to path
    plot_run_info(path, rewards)

    # Render the obtained final policy from training
    env = RenderOneAgentEnvironment(env)
    state, info = env.reset(seed=reset_key)
    while True:
        state, reward, terminated, truncated, _ = env.step(actor(state))
        if terminated or truncated:
            break
    env.render(path)

# TODO: Figure out a way to clearly seperate the structure needed for Experiment 2 (no obs),3 (obs) and 4 (obs and comm via reward)
def ddpg_train_patch_n_agents(env, num_episodes):
    path = create_exp_folder("Experiment2")
    episodes = list(range(1,num_episodes+1))
    action_dim, a_range = env.get_action_space()
    # Train agent
    (rewards, social_welfare), actors, critics, reset_key = n_agents_train_ddpg(env, episodes[-1], lr_c=1e-3, lr_a=2e-4, tau=0.005, action_dim=action_dim, state_dim=env.get_state_space()[1], action_max=a_range[1], hidden_dim=128, batch_size=128, seed=0, reset_seed=0)
    
    # Plot and save rewards figure to path
    plot_run_info(path, rewards, social_welfare, env.sw_fun)
    
    # Render the obtained final policy from training
    n_agents = env.get_num_agents()
    env = RenderNAgentsEnvironment(env)
    states, info = env.reset(seed=reset_key)
    while True:
        actions = [jnp.array(actors[i](states[i])) for i in range(n_agents)]
        states, rewards, terminated, truncated, _ = env.step(*actions)
        if np.all(terminated) or truncated:
            break
    env.render(path=path)

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
def experiment1():
    pass

"""
For this experiment we test the two-agent one-patch environment
The agents don't observe each other, and do not communicate.
Later I will extend this to multiple runs and use those to generate statistics for significance testing
"""
def experiment2():
    pass

"""
For this experiment we test the two-agent one-patch environment
The agents observe each other, but do not communicate.
Later I will extend this to multiple runs and use those to generate statistics for significance testing
"""
def experiment3():
    pass

"""
For this experiment we test the single-agent one-patch environment
The agents observe each other, and communicate via a social welfare function provided as a reward signal.
Later I will extend this to multiple runs and use those to generate statistics for significance testing
"""
def experiment4():
    pass

"""
For this experiment we test the single-agent one-patch environment
The agents observe each other, and communicate via by providing a message in addition to an action for their policy.
We will use the messages as state inputs, to train the critic on
Later I will extend this to multiple runs and use those to generate statistics for significance testing
"""
def experiment4():
    pass


if __name__ == "__main__":
    num_episodes = 100
    num_runs = 5
    
    # Uncomment the environment needed below
    env = NAgentsEnv(patch_radius=1, step_max=400, alpha=0, beta=0.5, e_init=10, n_agents=1, obs_others=False)
    #env = OneAgentEnv(patch_radius=0.5, step_max=400, alpha=2)
    
    # Uncomment the method needed below
    ddpg_train_patch_n_agents(env, num_episodes)
    #ddpg_train_patch(env, num_episodes)
    #wandb_ddpg_train_patch(env, num_episodes, num_runs=num_runs, hidden_dim=256, batch_size=100, warmup_steps=200)
    
    # Fill in the path of the policy and uncomment the method below it
    #path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policies", "e4nii8kg", "efficient-sweep-4")
    #patch_test_saved_policy(env, path, hidden_dim=256)