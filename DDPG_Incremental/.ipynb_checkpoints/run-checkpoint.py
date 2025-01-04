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

def plot_run_info(path, rewards, critics_loss_stats, actors_loss_stats, sw_fun=lambda x:0):
    x_range = list(range(rewards.shape[0]))
    c_range = np.linspace(0.0,1.0,rewards.shape[1])
    # Plot and save return
    plt.figure()
    if rewards.shape[1] == 1:
        plt.title("Return over episodes")
        plt.ylabel("Return")
    else:
        plt.title("Mean return across agents")
        plt.ylabel("Mean return")
    plt.fill_between(x_range, np.min(rewards[:,:,0], axis=1), np.max(rewards[:,:,0], axis=1), color='r', alpha=0.1)
    plt.plot(np.mean(rewards[:,:,0], axis=1), c='r') # Assumes the rewards are positive
    plt.xlabel("Episode")
    plt.savefig(os.path.join(path, "agent_episodes_return.png"))
    # Plot and save social welfare
    if sw_fun.__name__ != "<lambda>":
        plt.figure()
        plt.title("Nash social welfare obtained through rewards")
        plt.xlabel("Episode")
        plt.ylabel(f"Social welfare ({sw_fun.__name__})")
        plt.plot(rewards[:,0,1])
        plt.savefig(os.path.join(path, "in_episode_welfare.png"))

    cmap = plt.cm.Set1
    c_list = [cmap(i) for i in range(cmap.N)]
    
    # Plot and save actor loss
    plt.figure()
    plt.title("Critic loss per agent")
    plt.xlabel("Episode")
    plt.ylabel("Critic loss")
    for i in range(critics_loss_stats.shape[2]):
        plt.fill_between(x_range, critics_loss_stats[:,1,i], critics_loss_stats[:,2,i], color=c_list[i], alpha=0.1)
        plt.plot(critics_loss_stats[:,0,i], label=f"Agent {i+1}", c=c_list[i])
    #plt.ylim([critics_avg_loss.min(), critics_avg_loss.max()])
    plt.legend()
    plt.savefig(os.path.join(path, "critics_loss.png"))

    # Plot and save critic loss
    plt.figure()
    plt.title("Actor loss per agent")
    plt.xlabel("Episode")
    plt.ylabel("Actor loss")
    for i in range(actors_loss_stats.shape[2]):
        plt.fill_between(x_range, actors_loss_stats[:,1,i], actors_loss_stats[:,2,i], color=c_list[i], alpha=0.1)
        plt.plot(actors_loss_stats[:,0,i], label=f"Agent {i+1}", c=c_list[i])
    #plt.ylim([actors_avg_loss.min(), actors_avg_loss.max()])
    plt.legend()
    plt.savefig(os.path.join(path, "actors_loss.png"))

def ddpg_train_patch_n_agents(env, num_episodes, seed=0, path=""):
    jax.config.update('jax_threefry_partitionable', True)
    episodes = list(range(1,num_episodes+1))
    action_dim, a_range = env.get_action_space()
    # Train agent
    (rewards, social_welfare), (actors, critics), (as_loss, cs_loss), reset_key = n_agents_train_ddpg(env, episodes[-1], lr_c=5e-4, lr_a=1e-4, tau=0.05, action_dim=action_dim, state_dim=env.get_state_space(), action_max=a_range[1], hidden_dim=[256,256], batch_size=256, seed=seed, reset_seed=seed)
    
    # Plot and save rewards figure to path
    plot_run_info(path, rewards, cs_loss, as_loss, env.sw_fun)
    
    # Render the obtained final policy from training
    n_agents = env.n_agents
    env = RenderNAgentsEnvironment(env)
    env_state, states = env.reset(seed=reset_key)
    while True:
        actions = [jnp.array(actors[i](states[i])) for i in range(n_agents)]
        env_state, states, rewards, terminated, truncated, _ = env.step(env_state, *actions)
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
def experiment1(num_episodes, num_runs):
    for i in range(num_runs):
        path = create_exp_folder("Experiment1")
        print(f"Run {i+1} has been started")
        env = NAgentsEnv(n_agents=1, seed=i)
        ddpg_train_patch_n_agents(env, num_episodes, seed=i, path=path)

"""
For this experiment we test the two-agent one-patch environment
The agents don't observe each other, and do not communicate.
Later I will extend this to multiple runs and use those to generate statistics for significance testing
"""
def experiment2(num_episodes, num_runs):
    for i in range(num_runs):
        path = create_exp_folder("Experiment2")
        print(f"Run {i+1} has been started")
        env = NAgentsEnv(n_agents=2, seed=i)
        ddpg_train_patch_n_agents(env, num_episodes, seed=i, path=path)

"""
For this experiment we test the two-agent one-patch environment
The agents observe each other, but do not communicate.
Later I will extend this to multiple runs and use those to generate statistics for significance testing
"""
def experiment3(num_episodes, num_runs):
    for i in range(num_runs):
        path = create_exp_folder("Experiment3")
        print(f"Run {i+1} has been started")
        env = NAgentsEnv(n_agents=2, obs_others=True, seed=i)
        ddpg_train_patch_n_agents(env, num_episodes, seed=i, path=path)

"""
For this experiment we test the two-agent one-patch environment
The agents observe each other, and communicate via a social welfare function provided as a reward signal.
Later I will extend this to multiple runs and use those to generate statistics for significance testing
"""
def experiment4(num_episodes, num_runs):
    for i in range(num_runs):
        path = create_exp_folder("Experiment4")
        env = NAgentsEnv(n_agents=2, obs_others=True, seed=i, sw_fun=nash_sw, reward_dim=2)
        ddpg_train_patch_n_agents(env, num_episodes, seed=i, path=path)

"""
For this experiment we test the single-agent one-patch environment
The agents observe each other, and communicate via by providing a message in addition to an action for their policy.
We will use the messages as state inputs, to train the critic on
Later I will extend this to multiple runs and use those to generate statistics for significance testing
"""
def experiment5(num_episodes, num_runs):
    for i in range(num_runs):
        path = create_exp_folder("Experiment5")
        env = NAgentsEnv(n_agents=2, obs_others=True, seed=i, comm_dim=2, sw_fun=nash_sw, reward_dim=2)
        ddpg_train_patch_n_agents(env, num_episodes, seed=i, path=path)


if __name__ == "__main__":
    # Experiments can be run below
    experiment5(10,1)
    
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