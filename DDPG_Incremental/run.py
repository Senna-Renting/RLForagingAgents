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

def create_exp_folder(exp_name, test=False):
    folder_name = "runs"
    if test:
        folder_name= "tests"
    timepoint = datetime.now().strftime("%d-%m-%Y %H%M%S")
    path = os.path.join(folder_name, exp_name, timepoint)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def save_metadata_readme(path, metadata):
    with open(os.path.join(path, "README.md"), "a") as f:
        f.write("# Run information\n\n")
        f.write("## Algorithm (DDPG) hyperparameters:\n\n")
        f.write(f"State dimension: {metadata["state_dim"]}\n\n")
        f.write(f"Action dimension: {metadata["action_dim"]}\n\n")
        a_max = metadata["action_max"]
        f.write(f"Action range [min, max]: [-{a_max}, {a_max}]\n\n")
        f.write(f"Size of each hidden layer: {', '.join(metadata["hidden_dims"])}\n\n")
        f.write(f"Tau: {metadata["tau"]}\n\n")
        f.write(f"Learning rate (actor): {metadata["lr_actor"]}\n\n")
        f.write(f"Learning rate (critic): {metadata["lr_critic"]}\n\n")
        f.write(f"Gamma ($\\gamma$): {metadata["gamma"]}\n\n")
        f.write("## Training parameters:\n\n")
        f.write(f"Algorithm name: {metadata["alg_name"]}\n\n")
        f.write(f"Seed: {metadata["seed"]}\n\n")
        f.write(f"Number of episodes: {metadata["n_episodes"]}\n\n")
        f.write(f"Batch size: {metadata["batch_size"]}\n\n")
        f.write(f"Warmup size: {metadata["warmup_size"]}\n\n")
        f.write(f"Proportion of welfare metric (p_welfare) used: {metadata["p_welfare"]} \n\n")
        f.write(f"Action step noise: {metadata["act_noise"]} \n\n")
        if metadata["alg_name"] == "Normal TD3":
            f.write(f"Target noise: {metadata["target_noise"]}\n\n")
            f.write(f"Noise clip: {metadata["noise_clip"]}\n\n")
            f.write(f"Policy delay: {metadata["policy_delay"]}\n\n")
        f.write("## Environment parameters:\n\n")
        f.write(f"Number of agents: {metadata["n_agents"]}\n\n")
        f.write(f"Range of x-axis: [0, {metadata["x_max"]}]\n\n")
        f.write(f"Range of y-axis: [0, {metadata["y_max"]}]\n\n")
        f.write(f"Maximum amount of steps allowed in environment (training horizon): {metadata["step_max"]}\n\n")
        f.write(f"Maximum allowed velocity: {metadata["v_max"]}\n\n")
        f.write(f"Patch radius: {metadata["patch_radius"]}\n\n")
        f.write(f"Initial resource amount in patch: {metadata["s_init"]}\n\n")
        f.write(f"Initial energy of agents: {metadata["e_init"]}\n\n")
        f.write(f"Severity of penalties ($\\alpha$): {metadata["alpha"]}\n\n")
        f.write("Resource differential equation used: $s_{t+1} = s_t\\cdot\\eta - s_t\\cdot\\gamma^2 - s_t\\cdot\\beta$\n\n")
        f.write(f"Rate of resource decay ($\\gamma$): {metadata["env_gamma"]}\n\n")
        f.write(f"Rate of consuming resources for agents ($\\beta$): {metadata["beta"]}\n\n")
        f.write(f"Rate of resource growth ($\\eta$): {metadata["eta"]}\n\n")
        f.write("## General remaining parameters:\n\n")
        f.write(f"Distance below which we allow information sharing: {metadata["obs_range"]}\n\n")
        has_obs = "Yes" if metadata["obs_others"] else "No"
        f.write(f"Observe other agents (our form of communication): {has_obs}\n\n")

def run_ddpg(env, num_episodes, train_fun, path, train_args=dict(), skip_vid=False):
    # Extract/define initial variables
    episodes = np.arange(1,num_episodes+1)
    action_dim, a_range = env.get_action_space()
    train_args = {
        "state_dim":env.get_state_space(),
        "action_dim":action_dim,
        "action_max":a_range[1],
        **train_args
    }
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
    

    # Generate README
    save_metadata_readme(path, metadata)
    
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

"""
For this experiment we test the single-agent one-patch environment
Later I will extend this to multiple runs and use those to generate statistics for significance testing
"""
def experiment1(num_episodes, num_runs, test=False):
    for i in range(num_runs):
        path = create_exp_folder("Experiment1", test=test)
        print(f"Run {i+1} has been started")
        env = NAgentsEnv(n_agents=1, alpha=0.1)
        train_args = dict(seed=i)
        run_ddpg(env, num_episodes, n_agents_ddpg, path, train_args, skip_vid=False)

"""
For this experiment we test the two-agent one-patch environment
The agents don't observe each other, and do not communicate.
Later I will extend this to multiple runs and use those to generate statistics for significance testing
"""
def experiment2(num_episodes, num_runs, test=False):
    for i in range(num_runs):
        path = create_exp_folder("Experiment2", test=test)
        print(f"Run {i+1} has been started")
        env = NAgentsEnv(n_agents=2)
        train_args = dict(seed=i)
        run_ddpg(env, num_episodes, n_agents_ddpg, path, train_args, skip_vid=False)

"""
For this experiment we test the two-agent one-patch environment
The agents observe each other, but do not communicate.
Later I will extend this to multiple runs and use those to generate statistics for significance testing
"""
def experiment3(num_episodes, num_runs, test=False):
    for i in range(num_runs):
        path = create_exp_folder("Experiment3", test=test)
        print(f"Run {i+1} has been started")
        env = NAgentsEnv(n_agents=2, obs_others=True)
        train_args = dict(seed=i)
        run_ddpg(env, num_episodes, n_agents_ddpg, path, train_args, skip_vid=True)

"""
For this experiment we test the two-agent one-patch environment
The agents observe each other, and communicate via a social welfare function provided as a reward signal.
Later I will extend this to multiple runs and use those to generate statistics for significance testing
"""
def experiment4(num_episodes, num_runs, test=False):
    for i in range(num_runs):
        path = create_exp_folder("Experiment4", test=test)
        env = NAgentsEnv(n_agents=2, obs_others=True, obs_range=8)
        train_args = dict(seed=i, p_welfare=0.0)
        run_ddpg(env, num_episodes, n_agents_ddpg, path, train_args, skip_vid=False)

"""
For this experiment we test the single-agent one-patch environment
The agents observe each other, and communicate via by providing a message in addition to an action for their policy.
We will use the messages as state inputs, to train the critic on
Later I will extend this to multiple runs and use those to generate statistics for significance testing
"""
def experiment5(num_episodes, num_runs, test=False):
    pass

"""
Two agents who don't observe each other but do observe their own state as well as the patch's state. They are given a global reward signal in the form of nash social welfare (product of their returns as computed from the sample batches)
"""
def experiment6(num_episodes, num_runs, test=False):
    pass

if __name__ == "__main__":
    # Experiments can be run below
    experiment1(80,1)