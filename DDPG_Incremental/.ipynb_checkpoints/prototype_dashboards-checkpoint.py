# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Circle
import os
# Constants we will use to test on
step_max = 400
n_episodes = 50
n_agents = 2
n_stats = 3
reward_dim = 1
has_comms = True
test_shape_network = ((9,64), (64,64), (64,2))
path = "tests"
# The test data used to build and test the plots
test_data_in_patch = np.random.choice([True, False], (n_episodes, step_max, n_agents), p=[0.5,0.5])
test_data_penalties = np.random.uniform(-0.2,0.2, (n_episodes, step_max, n_agents, 1+int(has_comms)))
test_data_weights = [np.random.uniform(-1,1, (n_episodes, step_max, n_agents, *shape)) for shape in test_shape_network]
test_data_returns = np.random.uniform(-0.05, 0.2, (n_episodes, step_max, n_agents, reward_dim))
test_data_resources = np.random.uniform(0,10, (n_episodes, step_max))
test_data_loss = np.random.uniform(0,1, (n_episodes, n_stats, n_agents))
test_data_loss[:,0,:] = np.mean(test_data_loss, axis=1)
test_data_loss[:,1,:] = np.min(test_data_loss, axis=1)
test_data_loss[:,2,:] = np.max(test_data_loss, axis=1)
test_patch_info = ([2.5,2.5], 0.5, np.random.uniform(0,10, (step_max,)))
test_agents_poss = np.random.uniform(0,5, (step_max, n_agents, 2))
test_env_shape = [5,5]

### Put test functions for plots below (constants defined globally may NOT be IMPLICITLY used in the functions described below)
#### Not Tested
def plot_weights(path, weights):
    pass

def plot_rewards(path, rewards, colors=plt.cm.Set1.colors):
    x_range = np.arange(0,rewards.shape[0])
    returns = np.sum(rewards, axis=1)
    n_agents = rewards.shape[2]
    n_axes = 1
    fig = plt.figure()
    # Plot and save social welfare when multiple agents in environment
    if n_agents > 1:
        n_axes = 2
        ax = plt.subplot(n_axes, 1, 1)
        ax.set_title("Nash social welfare obtained through rewards")
        ax.set_xlabel("Episode")
        ax.set_ylabel(f"NSW")
        nsw_returns = np.prod(returns, axis=1)
        ax.plot(nsw_returns)
    # Plot and save return
    ax = plt.subplot(n_axes, 1, n_axes)
    ax.set_title("Return over episodes")
    ax.set_ylabel("Return")
    returns = np.sum(rewards, axis=1)
    [ax.plot(returns[:,i_a,0], c=colors[i_a], label=f"$A_{i_a+1}$") for i_a in range(n_agents)]
    ax.set_xlabel("Episode")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(path, "agent_episodes_return.png"))
    plt.close(fig)

#### Tested
"""
The agent_state and the patch_state of the NAgentsEnv class are used as input here
agent_state's shape: [n_episodes, step_max, n_agents, dim(x,y,x_dot,y_dot,e)]
patch_info's shape: ([dim(x,y,r)], [n_episodes, step_max, dim(s))])
"""
def plot_env(path, env_shape, patch_info, agents_state):
    n_episodes, step_max, n_agents, *_ = agents_state.shape
    patch_energy = patch_info[1]
    agent_size = env_shape[0]/100
    s_max = np.max(patch_info[1][-1])
    patch_pos = patch_info[0][:2]
    patch_radius = patch_info[0][2]
    agent_pos = lambda frame, i_a: agents_state[int(frame/step_max),frame%step_max, i_a, :2]
    norm = lambda frame: patch_energy[int(frame/step_max), frame%step_max,0]/s_max
    patch_color = lambda norm: (0.2,0.3+0.7*norm,0.2)
    fig = plt.figure()
    ax = plt.subplot(1,1,1)
    ax.set_xlim([0,env_shape[0]])
    ax.set_ylim([0,env_shape[1]])
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    patch = ax.add_patch(plt.Circle(patch_pos, patch_radius, color=patch_color(norm(0))))
    agents = [ax.add_patch(plt.Circle(agent_pos(0, i_a), agent_size, color='r')) for i_a in range(n_agents)]
    episode_text = ax.text(0.05,0.1, f"Episode: 1/{n_episodes}", transform=ax.transAxes)
    frame_text = ax.text(0.05,0.05, f"Timestep: 1/{step_max}", transform=ax.transAxes)
    plt.tight_layout()
    def update(frame):
        frame_text.set_text(f"Timestep: {(frame%step_max)+1}/{step_max}")
        episode_text.set_text(f"Episode: {int(frame/step_max)+1}/{n_episodes}")
        patch.set(color = patch_color(norm(frame)))
        for i_a, agent in enumerate(agents):
            agent.set(center = agent_pos(frame,i_a))
    fps = 24
    anim = FuncAnimation(fig, update, n_episodes*step_max, interval=1000/fps)
    anim.save(os.path.join(path, "runs_in_environment.mp4"))
    plt.show()
    plt.close(fig)

def plot_loss(path, name, data, colors=plt.cm.Set1.colors):
    n_episodes, n_stats, n_agents = data.shape
    x_range = np.arange(0,n_episodes)
    fig = plt.figure()
    plt.title(f"{name} loss of agents")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    for i_a in range(n_agents):
        plt.fill_between(x_range, data[:,1,i_a], data[:,2,i_a], color=colors[i_a], alpha=0.4)
        plt.plot(data[:,0,i_a], label=f"$A_{i_a+1}$", color=colors[i_a])
    plt.legend(loc="lower right")
    fig.savefig(os.path.join(path, f"{name}_loss.png"))
    plt.close(fig)
        
def plot_final_states_env(path, is_in_patch, patch_info, agents_states, rewards, colors=plt.cm.Set1.colors):
    step_max, n_agents, *_ = agents_states.shape
    agents_energy = agents_states[:,:,4]
    patch_ss = patch_info[1][-1]
    fig = plt.figure(figsize=(10,12))
    ax1 = plt.subplot(3,1,1)
    plt.title("Energy of agents")
    plt.ylabel("Energy")
    plt.xlabel("Timestep")
    [plt.plot(agents_energy[:,i_a], label=f"$A_{i_a+1}$", color=colors[i_a]) for i_a in range(n_agents)]
    make_colorbar = mark_patch_events(is_in_patch, ax=ax1)[1]
    ax1.legend(loc="lower right")
    ax2 = plt.subplot(3,1,2)
    plt.title("Rewards obtained")
    plt.ylabel("Reward")
    plt.xlabel("Timestep")
    [plt.plot(rewards[:,i_a], label=f"$A_{i_a+1}$", color=colors[i_a]) for i_a in range(n_agents)]
    mark_patch_events(is_in_patch, ax=ax2)
    ax2.legend(loc="lower right")
    ax3 = plt.subplot(3,1,3)
    plt.title("Resources in patch")
    plt.ylabel("Resource")
    plt.xlabel("Timestep")
    plt.plot(patch_ss, color=colors[2])
    mark_patch_events(is_in_patch, ax=ax3)
    make_colorbar()
    plt.tight_layout()
    fig.savefig(os.path.join(path, "final_states.png"))
    plt.close(fig)
    
# Works for two agents only
def mark_patch_events(is_in_patch, colors=plt.cm.Set1.colors, step=-1, ax=plt):
    n_episodes, step_max, n_agents = is_in_patch.shape
    encoding = np.arange(1,n_agents+1).reshape(-1,1)
    event_coding = np.sum(np.dot(is_in_patch[step, :, :],encoding), axis=1)[np.newaxis, :]
    # Reshape limits of y for event bar
    y_range = ax.get_ylim()
    y_length = y_range[1] - y_range[0]
    y_range = ax.set_ylim(y_range[0]-0.3*y_length, y_range[1])
    y_length = y_range[1] - y_range[0]
    x_range = ax.set_xlim(0,step_max)
    labels = ["None in patch", "$A_1$ in patch", "$A_2$ in patch", "All in patch"]
    l_colors = [colors[2],colors[0],colors[1],colors[3]]
    ticks = np.arange(0,4)
    if n_agents == 1:
        labels = labels[:2]
        l_colors = l_colors[:2]
        ticks = ticks[:2]
    fmt = FuncFormatter(lambda x, pos: labels[int(x)])
    cmap = ListedColormap(l_colors)
    image = ax.imshow(event_coding, extent=[x_range[0],x_range[1],y_range[0],y_range[1] - 0.8*y_length], aspect='auto', cmap=cmap, interpolation='nearest', vmin=ticks[0], vmax=ticks[-1])
    make_colorbar = lambda: plt.colorbar(image, ax=ax, format=fmt, ticks=ticks)
    return image, make_colorbar

def update_patch_events(is_in_patch, frame, image, ax=plt):
    n_episodes, step_max, n_agents = is_in_patch.shape
    encoding = np.arange(1,n_agents+1).reshape(-1,1)
    event_coding = np.sum(np.dot(is_in_patch[frame, :, :],encoding), axis=1)[np.newaxis, :]
    image.set_array(event_coding)

def plot_penalty(path, is_in_patch, data, name, colors=plt.cm.Set1.colors, bins=20):
    n_episodes, step_max, n_agents = data.shape
    # Penalty across episodes
    fig = plt.figure()
    plt.title(f"average of {name} penalty")
    plt.xlabel("Episode")
    plt.ylabel("Magnitude")
    [plt.plot(np.mean(data[:,:,i_a], axis=1), label=f"$A_{i_a+1}$", color=colors[i_a]) for i_a in range(n_agents)]
    plt.legend(loc="lower right")
    fig.savefig(os.path.join(path, f"{name}_penalty_over_episodes.png"))
    plt.close(fig)
    # Penalty within episodes
    bins = np.histogram(data, bins=bins)[1]
    text_box = dict(facecolor="green", alpha=0.5, edgecolor="darkgreen")
    text_object = lambda ax: plt.text(0.025,0.875,f"Episode 1/{n_episodes}", transform=ax.transAxes)
    fig = plt.figure()
    ax1 = plt.subplot(2,1,1)
    plt.title(f"Timeseries plot of {name} penalty")
    episode_num1 = text_object(ax1)
    episode_num1.set_bbox(text_box)
    plt.xlabel("Timestep")
    plt.ylabel("Magnitude")
    y_max = np.max(data)
    y_min = np.min(data)
    ax1.set_ylim([y_min, y_max])
    ax1.set_xlim([0,step_max])
    image, make_colorbar = mark_patch_events(is_in_patch, colors=colors, step=0, ax=ax1)
    make_colorbar()
    plots = [ax1.plot(data[0,:,i_a], label=f"$A_{i_a+1}$", color=colors[i_a])[0] for i_a in range(n_agents)]
    plt.legend(loc="lower right")
    ax2 = plt.subplot(2,1,2)
    plt.title(f"Histogram of {name} penalty")
    episode_num2 = text_object(ax2)
    episode_num2.set_bbox(text_box)
    hists = [ax2.hist(data[0,:,i_a], label=f"$A_{i_a+1}$", bins=bins, color=colors[i_a], alpha=0.5) for i_a in range(n_agents)]
    plt.xlabel("Magnitude")
    plt.ylabel("Count")
    plt.legend(loc="lower right")
    plt.tight_layout()
    def update(frame):
        ep_text = f"Episode {frame+1}/{n_episodes}"
        episode_num1.set_text(ep_text)
        episode_num2.set_text(ep_text)
        update_patch_events(is_in_patch, frame, image, ax=ax1)
        lines = [plot.set_ydata(data[frame,:,i_a]) for i_a,plot in enumerate(plots)]
        for i_a,hist in enumerate(hists):
            bar_container = hist[2]
            n, _ = np.histogram(data[frame,:,i_a], bins=bins)
            for count, rect in zip(n, bar_container.patches):
                rect.set_height(count)
    anim = FuncAnimation(fig, update, n_episodes)
    anim.save(filename=os.path.join(path, f"{name}_penalty.gif"))
    plt.close(fig)
    

if __name__ == "__main__":
    # Put any function of this file here to individually test/assess them
    #print(np.where(test_data_in_patch[1,:,0]))
    plot_penalty(path, test_data_in_patch, test_data_penalties[:,:,:,1], "communication")
    #plot_final_states_env(path, test_data_in_patch, test_data_returns[-1], test_data_resources[-1], test_data_returns[-1])
    #plot_loss(path, "critic", test_data_loss)
    #plot_env(path, test_env_shape, test_patch_info, test_agents_poss)