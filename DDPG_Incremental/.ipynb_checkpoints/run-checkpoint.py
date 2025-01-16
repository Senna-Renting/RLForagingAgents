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

# cmap: RdYlGn
# Function that converts saved weights (shape: n_episodes*n_agents*nn_shape) into a gif using plt and FuncAnimation
# TODO: 1. Turn weights into single figure plot
#       2. Animate figure through episodes
#       3. Draw arrows from output to input
def make_weights_gif(path, name, weights, exp_num=1):
    n_layers = len(weights)
    n_agents = weights[0].shape[1]
    n_episodes = weights[0].shape[0]
    # For now this is hardcoded maybe later connect this information to the networks (last element only included for Critic input)
    input_orderings = [(("Patch", [0,4]), ("Agent", [4,9]), ("Action", [9,11])),
                       (("Patch", [0,4]), ("Agent", [4,9]), ("Action", [9,11])),
                       (("Patch", [0,4]), ("Agent1", [4,9]), ("Agent2", [9,14]), ("Action", [14,16])),
                       (("Patch", [0,4]), ("Agent1", [4,9]), ("Agent2", [9,14]), ("Action", [14,16])),
                       (("Patch", [0,4]), ("Agent1", [4,9]), ("Agent2", [9,14]), ("Communication", [14,15]), ("Action", [15,17]))]
    input_ordering = input_orderings[exp_num-1]
    if name == "Actor":
        input_ordering = input_ordering[:-1]
    
    def add_ranking(data, axes=None):
        out_axes = [None]*(len(input_ordering)*2+1)
        input_sum = np.sum(data, axis=1)
        abs_sum = np.array([np.sum(np.abs(input_sum[slice[0]:slice[1]])) for (name, slice) in input_ordering])
        abs_sum /= np.max(abs_sum)
        ranking = abs_sum.argsort()
        fraction = 1/ranking.shape[0]
        draw_bar = lambda i,rank,axis: axis.fill_between([0.5, 0.5+0.1*abs_sum[rank]], i*fraction, (i+1)*fraction, color='g', clip_on=False)
        for i,rank in enumerate(ranking):
            text = f"{ranking.shape[0] - i}. {input_ordering[rank][0]} :"
            if axes is None:
                out_axes[0] = plt.gcf().add_axes((0.4, 0.7, 0.15, 0.15))
                out_axes[0].set_title("Ranking of which input \n changed the most", fontsize=10)
                out_axes[0].set_xlim([0,1])
                out_axes[0].set_ylim([0,1])
                out_axes[1+2*i+1] = draw_bar(i,rank,out_axes[0])
                out_axes[1+2*i] = out_axes[0].text(0,fraction*i,text, size=10)
                out_axes[1+2*i].set_bbox(dict(alpha=0))
            else:
                dummy = draw_bar(i,rank,axes[0])
                dp = dummy.get_paths()[0].vertices
                dummy.remove()
                axes[1+2*i].set_text(text)
                axes[1+2*i+1].set_paths([dp])
        return out_axes            
        
    # Only create a gif for runs with more than 1 episode
    if n_episodes > 1:
        for i_a in range(n_agents):
            fig = plt.figure()
            plt.suptitle(f"{name} gradient updates of Agent {i_a}")
            images = [None]*n_layers
            first_diff = weights[0][1,i_a] - weights[0][0,i_a]
            width_hidden = weights[0].shape[3]
            for i in range(n_layers):
                plt.subplot(1, n_layers, i+1)
                final_w = weights[i][1,i_a] - weights[i][0,i_a]
                all_w_grad = weights[i][1:,i_a] - weights[i][:-1,i_a]
                abs_max = np.max(np.abs(all_w_grad))
                # Make tall images (short end on x-axis)
                flip_axis = final_w.shape[0] < final_w.shape[1]
                if flip_axis:
                   final_w = final_w.T
                plt.xlabel("Input"*flip_axis + "Output"*(1-flip_axis))
                plt.ylabel("Output"*flip_axis + "Input"*(1-flip_axis))
                plt.xlim([0,final_w.shape[1]])
                plt.ylim([0,final_w.shape[0]])
                images[i] = plt.imshow(final_w, cmap='RdYlGn', vmin=-abs_max, vmax=abs_max, interpolation='nearest', extent=[0,final_w.shape[1],final_w.shape[0],0])
                layer_name = (i==0)*"Input" + (i==(n_layers-1))*"Output" + (i>0 and i!=(n_layers-1))*"Hidden"
                plt.title(f"{layer_name} layer")
                # Set minor ticks and gridlines
                ax = plt.gca()
                ax.set_yticks(np.arange(0, final_w.shape[0], 1), minor=True)
                ax.set_xticks(np.arange(0, final_w.shape[1], 1), minor=True)
                ax.grid(which="both", color='k', linestyle='-', linewidth=.5, alpha=0.1)
                ax.tick_params(which="minor", length=0)
            
            plt.subplot(1, n_layers, 2)
            plt.colorbar(shrink=0.3)
            episode_text = plt.gcf().text(0.4,0.2,f"Episode {1}/{n_episodes}")
            
            axes = add_ranking(first_diff)
            def update(frame):
                first_diff = weights[0][frame,i_a] - weights[0][frame-1,i_a]
                episode_text.set_text(f"Episode {frame+1}/{n_episodes}")
                add_ranking(first_diff, axes=axes)
                for i in range(n_layers):
                    final_w = weights[i][frame,i_a] - weights[i][frame-1, i_a]
                    flip_axis = final_w.shape[0] < final_w.shape[1]
                    if flip_axis:
                       final_w = final_w.T
                    images[i].set_array(final_w)
            anim = FuncAnimation(fig=fig, func=update, frames=n_episodes, interval=500)
            anim.save(filename=os.path.join(path, f"{name}_weights_agent{i_a}.gif"))
    

def plot_run_info(path, rewards, critics_loss_stats, actors_loss_stats, agents_info):
    (penalties, is_in_patch) = agents_info
    x_range = list(range(rewards.shape[0]))
    n_agents = rewards.shape[2]
    # Plot and save return
    plt.figure()
    if n_agents == 1:
        plt.title("Return over episodes")
        plt.ylabel("Return")
    else:
        plt.title("Mean return across agents")
        plt.ylabel("Mean return")
    returns = np.sum(rewards, axis=1)
    plt.fill_between(x_range, np.min(returns[:,:,0], axis=1), np.max(returns[:,:,0], axis=1), color='r', alpha=0.1)
    plt.plot(np.mean(returns[:,:,0], axis=1), c='r') # Assumes the rewards are positive
    plt.xlabel("Episode")
    plt.savefig(os.path.join(path, "agent_episodes_return.png"))
    # Plot and save social welfare when multiple agents in environment
    if n_agents > 1:
        plt.figure()
        plt.title("Nash social welfare obtained through rewards")
        plt.xlabel("Episode")
        plt.ylabel(f"NSW")
        plt.plot(np.prod(returns[:,:,0], axis=1))
        plt.savefig(os.path.join(path, "in_episode_welfare.png"))

    cmap = plt.cm.Set1
    c_list = [cmap(i) for i in range(cmap.N)]

    # Plot and save penalty vector
    plot_penalty(path, is_in_patch, penalties[:,:,:,0], "action")
    if penalties.shape[3] == 2:
        plot_penalty(path, is_in_patch, penalties[:,:,:,1], "communication")
    
    # Plot and save actor loss
    plot_loss(path, "critic", critics_loss_stats)

    # Plot and save critic loss
    plot_loss(path, "actor", actors_loss_stats)


def ddpg_train_patch_n_agents(env, num_episodes, seed=0, path="", exp_num=1):
    jax.config.update('jax_threefry_partitionable', True)
    episodes = list(range(1,num_episodes+1))
    action_dim, a_range = env.get_action_space()
    # Train agent
    rewards, ((actors, a_weights), (critics, c_weights)), (as_loss, cs_loss), agents_info, reset_key = n_agents_train_ddpg(env, episodes[-1], lr_c=1e-3, lr_a=2e-4, tau=0.01, action_dim=action_dim, state_dim=env.get_state_space(), action_max=a_range[1], hidden_dim=[64,64], batch_size=400, seed=seed, reset_seed=seed)
    (penalties, is_in_patch) = agents_info
    # Plot and save rewards figure to path
    plot_run_info(path, rewards, cs_loss, as_loss, agents_info)
    #make_weights_gif(path, "Actor", a_weights, exp_num=exp_num)
    #make_weights_gif(path, "Critic", c_weights, exp_num=exp_num)
    
    # Render the obtained final policy from training
    n_agents = env.n_agents
    env_state, states = env.reset(seed=reset_key)
    patch_states = np.empty((env.step_max, *env_state[1].shape))
    agent_states = np.empty((env.step_max, *env_state[0].shape))
    for i in range(env.step_max):
        actions = [jnp.array(actors[i](states[i])) for i in range(n_agents)]
        env_state, states, _, terminated, truncated, __ = env.step(env_state, *actions)
        (agents_state, patch_state,step_max) = env_state
        agent_states[i] = agents_state
        patch_states[i] = patch_state
        if np.all(terminated) or truncated:
            break
    plot_final_states_env(path, is_in_patch, patch_states, agent_states, rewards[-1])
    plot_env(path, env.size(), patch_states, agent_states)
    

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