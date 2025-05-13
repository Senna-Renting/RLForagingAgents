import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.collections import LineCollection
from matplotlib.ticker import FuncFormatter
import os
import json
from run import run_actor_test


COLORS = plt.cm.Set1.colors

"""
Plots the Kullback-Leibler divergence between the learned critic and the target critic for each timestep across agents
"""
def plot_kl(path, data, colors=COLORS):
    fig = plt.figure()
    plt.title(f"Kullback-Leibler distance for Actor vs batch actions")
    plt.xlabel("Timestep")
    plt.ylabel("KL(a, $\\hat{a}$)")
    [plt.plot(data[:,i_a], label=f"Agent {i_a+1}", color=colors[i_a]) for i_a in range(data.shape[1])]
    plt.legend()
    fig.savefig(os.path.join(path, f"kl_critics.png"))
    plt.close()

"""
Plots a histogram of the learned and target critic values seen in the last episode
"""
def plot_cvals(path, data):
    for i_a in range(data.shape[1]):
        fig = plt.figure()
        plt.title(f"Distribution of Q-values for Agent {i_a+1}")
        plt.xlabel("Q(s,a)")
        plt.ylabel("Amount")
        d_range = (data[:,i_a,:].min(),data[:,i_a,:].max())
        plt.hist(data[:,i_a,1], label="$Q_{target}$", bins=100, alpha=0.5, range=d_range)
        plt.hist(data[:,i_a,0], label=f"$Q$", bins=100, alpha=0.5, range=d_range)
        plt.legend()
        fig.savefig(os.path.join(path, f"critic_values_agent{i_a+1}.png"))
        plt.close()
        

"""
This function should plot the average, minimum and maximum return of the runs, inside a given folder across their episodes.
The resulting plot will be put inside the path folder.
Runs should be of equal length when running this function.

Input:
    path: location of folder containing runs on device
    window: if greater than 1 we generate a moving average of the return
Return: (min, mean, max)
"""
def get_grouped_return(path, window=1):
    return_data = []
    num_episodes = 0
    # Get folders of seperate runs
    for run in os.listdir(path): 
        run_p = os.path.join(path, run)
        if os.path.isdir(run_p) and os.path.exists(os.path.join(run_p, "metadata.json")):
            # For each run get the return data
            path_return_data = os.path.join(run_p, "data", "returns.dat")
            path_metadata = os.path.join(run_p, "metadata.json")
            with open(path_metadata, 'r') as f:
                metadata = json.load(f)
                num_episodes = metadata["n_episodes"]
                return_shape = (num_episodes, metadata["step_max"], metadata["n_agents"])
                run_data = np.memmap(path_return_data, mode="r+", shape=return_shape, dtype="float32").copy()
                mean_return = np.mean(np.sum(run_data, axis=1),axis=1)
                return_data.append(mean_return)
    # Convert return data list into an array
    return_data = np.array(return_data)
    avg_window = np.full((window,), 1/window)
    min = np.convolve(return_data.min(axis=0), avg_window, 'valid')
    max = np.convolve(return_data.max(axis=0), avg_window, 'valid')
    mean = np.convolve(return_data.mean(axis=0), avg_window, 'valid')
    # Generate the plot
    x_range = np.arange(mean.shape[0])
    fig = plt.figure()
    plt.title(f"Average return of agents over episodes")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.fill_between(x_range, min, max, color="r", alpha=0.4, label="min, max bound")
    plt.plot(x_range, mean, color="b", label="Mean(A1, A2)")
    plt.grid()
    plt.legend(loc="lower right")
    fig.savefig(os.path.join(path, f"average_NSW.png"))
    plt.close(fig)
    return (min, mean, max)

def exp1_plots(out_path, *paths, colors=COLORS, window=4):
    # Plot data and save to png
    fig = plt.figure()
    plt.title("Comparing learning curves with communication")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    for i,path in enumerate(paths):
        label = os.path.basename(path)
        min, mean, max = get_grouped_return(path, window)
        x_range = np.arange(0, mean.shape[0])
        plt.fill_between(x_range, min, max, color=colors[i], alpha=0.2)
        plt.plot(mean, color=colors[i], label=label)
    plt.grid()
    plt.legend()
    fig.savefig(os.path.join(out_path, "experiment1_result.png"))

"""
This function derives the data needed for RQ1 plots, based on the stored data
"""
def rq1_data(patch_info, agents_state, actions):
    patch_position = patch_info[0][:2]
    energy = agents_state[:,:,:,4]
    positions = agents_state[:,:,:,:2]
    dist_agents = np.sqrt(np.sum(np.power(np.diff(positions, axis=2),2),axis=3))[:,:,0]
    # We compute the distance of each agent w.r.t the edge of the patch
    dist_agents_patch = np.sqrt(np.sum(np.power(positions - patch_position[np.newaxis,np.newaxis,np.newaxis,:],2),axis=3)) - patch_info[0][2]
    if actions.shape[3] > 2:
        communication = actions[:,:,:,2]
        attention_other = np.flip(actions[:,:,:,3],axis=2)
        comm_filter = (communication > 0.5) & (attention_other > 0.5)
    else: 
        comm_filter = None
    return energy, dist_agents, dist_agents_patch, comm_filter

""" 
Main function for generating information gained from each episode
"""
def episode_results(path, energy, dist_agents, dist_agents_patch, comm_filter, plot_env=lambda episode,path: None):
    path = os.path.join(path, "rq1_plots")
    os.mkdir(path)
    for episode in range(energy.shape[0]):
        sub_path = os.path.join(path, f"episode {episode}")
        os.mkdir(sub_path)
        rq1_plots_per_episode(episode, sub_path, energy, dist_agents, dist_agents_patch, comm_filter, colors=COLORS)
    # Only make a video for the final episode
    plot_env(episode, sub_path)

"""
Subfunction for generating the plots of a single episode for RQ1
"""
def rq1_plots_per_episode(episode, path, energy, dist_agents, dist_agents_patch, comm_filter, colors=COLORS):
    # First we implement the plots on the last run, later we can make folders for each successive run and it's results
    energy, dist_agents, dist_agents_patch = energy[episode, :, :], dist_agents[episode, :], dist_agents_patch[episode,:,:]
    x_range = np.arange(0,energy.shape[0])
    
    # The code for the plots
    fig2 = plot_dist_agents(dist_agents, x_range)
    if comm_filter is None:
        fig1 = plot_internal_energy(energy, x_range)
        fig3 = plot_dist_agents_patch(dist_agents_patch, x_range)
    else:
        comm_filter = comm_filter[episode, :, :]
        fig1 = plot_internal_energy_comm(energy, x_range, comm_filter)
        fig3 = plot_dist_agents_patch_comm(dist_agents_patch, x_range, comm_filter)
    
    # Save figures to PNG
    fig1.savefig(os.path.join(path, "internal_energy.png"))
    fig2.savefig(os.path.join(path, "dist_agents.png"))
    fig3.savefig(os.path.join(path, "dist_agents_patch.png"))
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)

def plot_comm_frequency(path, comm_filter, colors=COLORS):
    fig1 = plt.figure()
    plt.title("Mean amount of communication steps over episodes")
    plt.xlabel("Episode")
    plt.ylabel("Communication steps")
    plt.plot(comm_filter.sum(axis=1).mean(axis=1), linewidth=2)
    fig2 = plt.figure()
    plt.title("Mean amount of communication events over episodes")
    plt.xlabel("Episode")
    plt.ylabel("Communication events")
    plt.plot(np.diff(comm_filter,axis=1).sum(axis=1).mean(axis=1), linewidth=2)
    fig1.savefig(os.path.join(path, "comm_steps.png"))
    fig2.savefig(os.path.join(path, "comm_events.png"))

def plot_dist_agents_patch_comm(dist_agents_patch, x_range, comm_filter, colors=COLORS):
    fig,ax = plt.subplots(figsize=(10,5))
    ax.set_title("Relation of distance to patch and communication")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Distance(Agent, Patch)")
    plt.plot([x_range[0],x_range[-1]],[0,0],label="Patch border", c='r', linestyle="--")
    for a_i in range(dist_agents_patch.shape[1]):
        cmap = plt.get_cmap('viridis')
        plt.scatter(x_range[::20], dist_agents_patch[::20,a_i], label=f"Agent {a_i}", color=colors[a_i])
        sm = plot_color_lines(x_range, dist_agents_patch[:,a_i], comm_filter[:,a_i], ax, cmap)
    cbar = plt.colorbar(sm, ax=ax, ticks=[0.25,0.75])
    cbar.ax.set_yticklabels(['No communication', 'Communication'])
    plt.legend()
    return fig

def plot_dist_agents_patch(dist_agents_patch, x_range, colors=COLORS):
    fig = plt.figure()
    plt.title("Relation of distance to patch")
    plt.xlabel("Timestep")
    plt.ylabel("Distance(Agent, Patch)")
    plt.plot([x_range[0],x_range[-1]],[0,0],label="Patch border", c='r', linestyle="--")
    for a_i in range(dist_agents_patch.shape[1]):
        plt.plot(dist_agents_patch[:,a_i], color=colors[a_i], linestyle="-", label=f"Agent {a_i}")
    plt.legend()
    return fig

def plot_internal_energy_comm(energy, x_range, comm_filter, colors=COLORS):
    fig,ax = plt.subplots(figsize=(10,5))
    ax.set_title("Relation between internal energy and communication")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Internal energy")
    for a_i in range(energy.shape[1]):
        cmap = plt.get_cmap('viridis')
        plt.scatter(x_range[::20], energy[::20,a_i], label=f"Agent {a_i}", color=colors[a_i])
        sm = plot_color_lines(x_range, energy[:,a_i], comm_filter[:,a_i], ax, cmap)
    cbar = plt.colorbar(sm, ax=ax, ticks=[0.25,0.75])
    cbar.ax.set_yticklabels(['No communication', 'Communication'])
    plt.legend()
    return fig

def plot_color_lines(x,y,c,ax,cmap=plt.get_cmap('viridis')):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    cmap = ListedColormap([cmap(0.0), cmap(1.0)], N=2)
    bounds = [-0.5, 0.5, 1.5]  # so 0 → bin 0, 1 → bin 1
    norm = BoundaryNorm(bounds, cmap.N)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # dummy array — colorbar only needs cmap + norm
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2)
    c = c.astype(int)
    lc.set_array(c[:-1])
    ax.add_collection(lc)
    ax.autoscale()
    return sm
        
        

def plot_internal_energy(energy, x_range, colors=COLORS):
    fig = plt.figure()
    plt.title("Internal energy of agents")
    plt.xlabel("Timestep")
    plt.ylabel("Internal energy")
    for a_i in range(energy.shape[1]):
        plt.plot(x_range, energy[:,a_i], color=colors[a_i], linestyle="-", label=f"Communication Agent {a_i}")
    plt.legend()
    return fig

def plot_dist_agents(dist_agents, x_range, colors=COLORS):
    fig = plt.figure()
    plt.title("Distance between agents per episode")
    plt.xlabel("Timestep")
    plt.ylabel("Distance(Agent1, Agent2)")
    plt.plot(x_range, dist_agents, colors[2])
    return fig


def env_vars_data(patch_info, agents_state, actions):
    n_agents = actions.shape[2]
    action_dim = actions.shape[-1]
    agents_data = []
    energy_state = np.reshape(agents_state[:,:,:,4], (-1, n_agents))
    resource_state = patch_info[1][:,:,-1].flatten()
    action_state = np.linalg.norm(np.reshape(actions[:,:,:,:2], (-1, n_agents, 2)), axis=2)
    agents_data.append(("Action", action_state))
    if action_dim >= 3:
        comms_state = np.reshape(np.max(actions[:,:,:,2:-1], axis=3), (-1, n_agents))
        agents_data.append(("Comms", comms_state))
    if action_dim >= 4:
        attention_state = np.reshape(actions[:,:,:,-1], (-1, n_agents))
        agents_data.append(("Attention", attention_state))
    agents_data.append(("Energy", energy_state))
    return resource_state, agents_data

def plot_env_vars(resource_state, agents_data, axes, trail=100, a_colors=COLORS):
    font_size = 11
    scale_y = lambda ax, data: ax.set_ylim([data.min()-data.max()*0.1, data.max()*1.1])
    resource = axes[0].plot(resource_state[0], c='g')[0]
    axes[0].set_ylabel("Patch", fontsize=font_size)
    axes[0].set_xlim([0,trail])
    scale_y(axes[0], resource_state)
    axes[0].tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
    agent_plots = []
    for i,(name,d) in enumerate(agents_data):
        agent_plots.append([axes[i+1].plot(d[0, i_a], c=a_colors[i_a])[0] for i_a in range(d.shape[-1])])
        scale_y(axes[i+1], d)
        axes[i+1].set_xlim([0,trail])
        axes[i+1].set_ylabel(name, fontsize=font_size)
        axes[i+1].tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
        if i == len(agents_data)-1:
            axes[i+1].set_xlabel("Time", fontsize=font_size)
    data = (resource_state, [d for name,d in agents_data])
    plots = (resource, agent_plots)
    return data, plots

def update_env_vars(frame, data, plots, trail=100):
    start = np.max([0, frame-trail])
    x_range = np.arange(0,frame-start)
    plots[0].set_data(x_range, data[0][start:frame])
    for i,d in enumerate(data[1]):
        for i_a in range(d.shape[-1]):
            plots[1][i][i_a].set_data(x_range, d[start:frame, i_a])

def plot_rewards(path, rewards, colors=COLORS):
    x_range = np.arange(0,rewards.shape[0])
    returns = np.sum(rewards, axis=1)
    n_agents = rewards.shape[2]
    n_axes = 1
    fig = plt.figure()
    # Plot and save return
    ax = plt.subplot(n_axes, 1, n_axes)
    ax.set_title("Return over episodes")
    ax.set_ylabel("Return")
    returns = np.sum(rewards, axis=1)
    [ax.plot(returns[:,i_a], c=colors[i_a], label=f"$A_{i_a+1}$") for i_a in range(n_agents)]
    ax.set_xlabel("Episode")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(path, "agent_episodes_return.png"))
    plt.close(fig)

def plot_final_welfare(path, returns, colors=COLORS):
    nsw = np.sqrt(np.prod(returns, axis=2)).sum(axis=1)
    fig = plt.figure()
    plt.plot(nsw)
    plt.title("Welfare of agents across episodes")
    plt.xlabel("Episode")
    plt.ylabel("$NSW(R_1, R_2)$")
    fig.savefig(os.path.join(path, "average_nsw_over_episodes.png"))
    plt.close(fig)

def plot_succes_rate_comm(path, actions):
    communication = actions[:,:,:,2]
    max_amount = communication.shape[1]*communication.shape[2]
    attention_other = np.flip(actions[:,:,:,3], axis=2)
    comm_amount = (communication*attention_other).sum(axis=1).sum(axis=1)/max_amount
    fig = plt.figure()
    plt.title("Percentage of communication per episode")
    plt.xlabel("Episode")
    plt.ylabel("Total %")
    plt.plot(comm_amount, linewidth=2)
    fig.savefig(os.path.join(path, "comm_amount.png"))

def plot_actor_returns(out_path, *paths, num_episodes=30):
    fig = plt.figure()
    plt.title("Compare return histograms")
    returns = np.array([run_actor_test(path, num_episodes)[2].mean(axis=2).sum(axis=1) for path in paths])
    for i,path in enumerate(paths):
        label = os.path.basename(os.path.dirname(path))
        plt.hist(returns[i], bins=30, range=(returns.min(), returns.max()), label=label, alpha=0.4)  
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    plt.legend()
    fig.savefig(os.path.join(out_path, "exp1_return_hists.png"))

"""
The agent_state and the patch_state of the NAgentsEnv class are used as input here
agent_state's shape: [n_episodes, step_max, n_agents, dim(x,y,x_dot,y_dot,e)]
patch_info's shape: ([dim(x,y,r)], [n_episodes, step_max, dim(s))])
"""
def plot_env(path, episode, env_shape, patch_info, agents_state, actions, a_colors=COLORS):
    a_colors=COLORS
    n_episodes, step_max, n_agents, *_ = agents_state.shape
    action_names = ["Horizontal acc", "Vertical acc", "Communication", "Attention"]
    patch_energy = patch_info[1]
    agent_size = env_shape[0]/70
    s_max = np.max(patch_info[1])
    patch_pos = patch_info[0][:2]
    patch_radius = patch_info[0][2]
    agent_pos = lambda frame, i_a: agents_state[int(frame/step_max),frame%step_max, i_a, :2]
    agent_radius = lambda frame: patch_energy[int(frame/step_max), frame%step_max,0]
    norm = lambda frame: patch_energy[int(frame/step_max), frame%step_max,-1]/s_max
    patch_color = lambda norm: (0.2,0.3+0.7*norm,0.2)
    fig = plt.figure()
    ax = fig.add_axes([0.05,0.05,0.5,1])
    ax.set_xlim([0,env_shape[0]])
    ax.set_ylim([0,env_shape[1]])
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    resource_state, agents_data = env_vars_data(patch_info, agents_state, actions)
    state_axes = [fig.add_axes([0.6,0.7-i*0.15,0.38,0.15]) for i in range(len(agents_data)+1)]
    data, plots = plot_env_vars(resource_state, agents_data, state_axes)
    start_frame = episode*step_max
    patch = ax.add_patch(plt.Circle(patch_pos, patch_radius, color=patch_color(norm(start_frame))))
    agents = [ax.add_patch(plt.Circle(agent_pos(start_frame, i_a), agent_size, facecolor=a_colors[i_a], linewidth=1, edgecolor=(0.9,0.9,0.9))) for i_a in range(n_agents)]
    episode_text = ax.text(0.05,0.1, f"Episode: 1/{n_episodes}", transform=ax.transAxes)
    frame_text = ax.text(0.05,0.05, f"Timestep: 1/{step_max}", transform=ax.transAxes)

    def update(frame):
        frame_text.set_text(f"Timestep: {frame+1}/{step_max}")
        episode_text.set_text(f"Episode: {episode+1}/{n_episodes}")
        radius = patch_radius if patch_energy.shape[-1] == 1 else agent_radius(frame)
        frame += episode*step_max
        patch.set(color = patch_color(norm(frame)), radius = radius)
        for i_a, agent in enumerate(agents):
            agent.set(center = agent_pos(frame,i_a))
        update_env_vars(frame, data, plots)
        return [frame_text, episode_text, patch, *agents, ]
    
    fps = 24
    anim = FuncAnimation(fig,update, step_max, init_func=lambda: update(0), interval=1000/fps, blit=True, cache_frame_data=False)
    anim.save(os.path.join(path, "env_result_episode.mp4"))
    plt.close(fig) 

def plot_loss(path, name, data, colors=COLORS):
    steps, n_agents = data.shape
    fig = plt.figure()
    if name == "critic":
        data = np.clip(data, 0, 100)
    plt.title(f"{name} loss of agents")
    plt.xlabel("Timestep")
    plt.ylabel(f"{name} loss")
    for i_a in range(n_agents):
        plt.plot(data[:,i_a], label=f"$A_{i_a+1}$", color=colors[i_a])
    plt.legend(loc="lower right")
    fig.savefig(os.path.join(path, f"{name}_loss.png"))
    plt.close(fig)
        
def plot_final_states_env(path, is_in_patch, patch_info, agents_states, rewards, colors=COLORS):
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
def mark_patch_events(is_in_patch, colors=COLORS, step=-1, ax=plt):
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

def plot_penalty(path, is_in_patch, data, name, colors=COLORS, bins=20):
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
    pass