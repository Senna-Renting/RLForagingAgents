import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.collections import LineCollection
from matplotlib.ticker import FuncFormatter
import os
import json
from run import run_actor_test, run_multi_actor_test
from environment import compute_NSW, energy_to_reward
from scipy.stats import ttest_ind, pearsonr
from seaborn import jointplot
from statsmodels.regression.linear_model import OLS

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
    fig = plt.figure()
    plt.title(f"Distribution of sampled Q-values at last timestep")
    plt.xlabel("Q(s,a)")
    plt.ylabel("Amount")
    d_range = (data[:,:].min(),data[:,:].max())
    plt.hist(data[:,1], label="$Q_{target}$", bins=100, alpha=0.5, range=d_range)
    plt.hist(data[:,0], label=f"$Q$", bins=100, alpha=0.5, range=d_range)
    plt.legend()
    fig.savefig(os.path.join(path, f"critic_values_agents.png"))
    plt.close()
        
"""
Function that groups data of a given type found in the given run folder.
Also by standard gives the metadata of the run folder.

Parameters:
- path: location of the run folder on the device
- filename: file to pull the data from

Returns: (grouped data, metadata)
"""
def get_grouped_data(path, filename):
    data = []
    # Get folders of seperate runs
    for run in os.listdir(path): 
        run_p = os.path.join(path, run)
        if os.path.isdir(run_p) and os.path.exists(os.path.join(run_p, "metadata.json")):
            # For each run get the return data
            data.append(get_data(run_p, filename))
    return np.asarray(data)

"""
Get data for a single run.

Parameters:
- path: location of the single run folder
- filename: file to pull the data from
"""
def get_data(path, filename):
    path_data = os.path.join(path, "data", filename)
    run_data = np.lib.format.open_memmap(path_data, mode="r")
    return run_data

def get_metadata(path):
    return json.load(open(os.path.join(path, "metadata.json"), "r"))

def get_grouped_comm(path, window=1):
    comm_data = get_grouped_data(path, "actions.npy")[:,:,:,:,2:]
    max_amount = comm_data.shape[2]*comm_data.shape[3]
    comm_filter = np.sqrt(comm_data[:,:,:,:,0]*np.flip(comm_data[:,:,:,:,1])).sum(axis=3).sum(axis=2)/max_amount
    # Convert return data list into an array
    avg_window = np.full((window,), 1/window)
    min = np.convolve(comm_filter.min(axis=0), avg_window, 'valid')
    max = np.convolve(comm_filter.max(axis=0), avg_window, 'valid')
    mean = np.convolve(comm_filter.mean(axis=0), avg_window, 'valid')
    # Generate the plot
    x_range = np.arange(mean.shape[0])
    fig = plt.figure()
    plt.title(f"Percentage of communication between agents")
    plt.xlabel("Episode")
    plt.ylabel("Communication %")
    plt.fill_between(x_range, min, max, color="r", alpha=0.4, label="min, max bound")
    plt.plot(x_range, mean, color="b", label="Mean(A1, A2)")
    plt.grid()
    plt.legend(loc="lower right")
    fig.savefig(os.path.join(path, f"comm_amount.png"))
    plt.close(fig)
    return (min, mean, max)

def get_cond(a, b):
    if b.sum() == 0:
        return 0
    return ((a) & (b)).sum() / b.sum()

def get_marg(a):
    return a.sum() / a.size

def get_trend(out_path, var_name, exp_name, num_episodes, *run_paths, multi_actor=False):
    # Get data
    data = []
    data2 = []
    probs = {
        "$p(a(t) > 0.7 | c(t) > 0)$": [],
        "$p(c(t) > 0.7 | a(t) > 0)$": []
    }
    keys = list(probs.keys())
    att = []
    comm = []
    c_strength = []
    collected = []
    collected_m = []
    collected_std = []
    x_range = []
    x_labels = {"p_welfare": "Proportion of social welfare: $p_{w}$", 
                "s_init": "Maximum resources allowed in patch ($\\rho (0)$)", 
                "p_comm": "Communication penalty: $p_{c} = p_{a}$"}
    runs = [ f.path for f in os.scandir(out_path) if f.is_dir()]
    if not multi_actor:
        run_actor = run_actor_test
    else:
        run_actor = run_multi_actor_test

    for path in runs:
        print(f"Starting test for run: {path}...")
        states, actions, eaten, rewards, metadata = run_actor(path, num_episodes)
        c = actions[:,:,:,2].flatten()
        a = np.flip(actions[:,:,:,3], axis=2).flatten()
        att.append(get_marg(actions[:,:,:,3] > 0.7))
        comm.append(get_marg(actions[:,:,:,2] > 0.7))
        nsw_returns = compute_NSW(energy_to_reward(states[:,:,:,4], metadata["e_max"]), axis=2).sum(axis=1)
        comm_filter = get_comm_filter(actions)
        nonzero_channels = ((actions[:,:,:,3] > 0) & (np.flip(actions[:,:,:,2],axis=2) > 0))
        x_range.append(metadata[var_name])
        r_std, r_mean = (nsw_returns.std(), nsw_returns.mean())
        data.append([r_mean-r_std, r_mean, r_mean+r_std])
        c_mean = comm_filter.sum()/nonzero_channels.sum() 
        data2.append(c_mean)
        c_means = get_comm_strength(actions).reshape(actions.shape[0], -1).mean(axis=1).flatten()
        c_strength.extend(c_means)
        collected.extend(eaten[:,-1].flatten())
        collected_m.append(eaten[:,-1].mean(axis=0))
        collected_std.append(eaten[:,-1].std(axis=0))
        probs[keys[0]].append(get_cond(actions[:,:,:,3].flatten() > 0.7, np.flip(actions[:,:,:,2], axis=2).flatten() > 0))
        probs[keys[1]].append(get_cond(actions[:,:,:,2].flatten() > 0.7, np.flip(actions[:,:,:,3], axis=2).flatten() > 0))

    sorted_idx = np.argsort(x_range)
    x_range = np.array(x_range)[sorted_idx]
    data = np.array(data)[sorted_idx]
    data2 = np.array(data2)[sorted_idx]
    comm = np.array(comm)[sorted_idx]
    att = np.array(att)[sorted_idx]
    collected_m = np.array(collected_m)[sorted_idx]
    collected_std = np.array(collected_std)[sorted_idx]
    probs = {key: np.array(value)[sorted_idx] for key, value in probs.items()}
    # Plot the data
    fig1 = plt.figure(figsize=(8,5))
    plt.title(f"Effect of {exp_name}")
    plt.plot(x_range, data[:,1], c='b')
    plt.fill_between(x_range, data[:,0], data[:,2], color="b", alpha=0.4, label="± 1 std")
    plt.scatter(x_range, data[:,1], c='b')
    plt.ylabel("Return")
    plt.xlabel(var_name)
    plt.grid()
    fig1.savefig(os.path.join(out_path, "comm_return_trend.png"))
    fig2 = plt.figure(figsize=(8,5))
    plt.title(f"Amount of communication for {exp_name}")
    plt.plot(x_range, data2, c='b', label=r"$p(\sqrt{a(t)\cdot c(t)} > 0.7 | a(t) \neq 0 \wedge c(t) \neq 0)$")
    plt.plot(x_range, comm, linestyle="--", color="r", label=r"$p(c(t) > 0.7)$")
    plt.plot(x_range, att, linestyle="--", color="g", label=r"$p(a(t) > 0.7)$")
    plt.scatter(x_range, data2, c='b')
    plt.ylabel("Proportion")
    plt.xlabel(x_labels[var_name])
    plt.legend()
    plt.grid()
    fig2.savefig(os.path.join(out_path, "comm_percent_trend.png"))
    
    # Plot the lines for analysis of effective communication
    fig3 = plt.figure()
    for i,(key,value) in enumerate(probs.items()):
        plt.plot(x_range, value, label=key)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel('Probability')
    plt.xlabel(x_labels[var_name])
    plt.title(f'Effectiveness of communication for {exp_name}')
    plt.legend(loc='upper right')
    plt.ylim(0, 1)
    plt.grid()
    fig3.savefig(os.path.join(out_path, "comm_effectiveness.png"))

    # Plot resource collection efficiency
    fig4 = plt.figure()
    plt.title(f"Resource collection efficiency for {exp_name}")
    plt.xlabel(x_labels[var_name])
    plt.ylabel("Collected resources")
    plt.errorbar(x_range, collected_m, yerr=collected_std, fmt='-o', color='b', label="Mean collected resources ± 1 std")
    plt.legend()
    plt.grid()
    fig4.savefig(os.path.join(out_path, "resource_collection_efficiency.png"))

    # Plot scatterplot of resource collection against communication
    collected = np.array(collected)[:,np.newaxis]
    res = OLS(c_strength, np.hstack((collected, np.ones_like(collected)))).fit()
    fig5 = plt.figure()
    plt.title(f"Resource collection vs communication for {exp_name}")
    plt.xlabel("Collected resources")
    plt.ylabel("Mean(Communication strength)")
    plt.scatter(collected, c_strength, c='b')
    plt.plot(collected, res.fittedvalues, label="Fitted line", color='r', linestyle='--')
    plt.plot([], [], ' ', label="$r^2$ = {:.2f}".format(res.rsquared))
    plt.plot([], [], ' ', label="y = {:.2f}x + {:.2f}".format(res.params[0], res.params[1]))
    plt.plot([], [], ' ', label="p values: [{:.2f}, {:.2f}]".format(res.pvalues[0], res.pvalues[1]))
    plt.legend()
    plt.grid()
    fig5.savefig(os.path.join(out_path, "resource_collection_vs_comm.png"))


    

def get_comm_strength(actions):
    a = actions[:,:,:,3]
    c = actions[:,:,:,2]
    return np.sqrt(np.flip(a, axis=2)*c)    

def get_derived_data(states, actions):
    a = actions[:,:,:,3]
    c = actions[:,:,:,2]
    energy = states[:,:-1,:,4]
    c_strength = get_comm_strength(actions)
    patch_pos = states[:,:-1,:,-4:-2]
    patch_radius = states[:,:-1,:,-2]
    d_to_patch = (np.sqrt(np.power(states[:,:-1,:,:2] - patch_pos, 2).sum(axis=3)) - patch_radius) # Euclidean distance
    return a, c, energy, c_strength, d_to_patch

"""
A group function for all plots that need to be created by
running the final actor in the environment for some episodes.

Parameters:
- path: path to save to and get the actor network from
- num_episodes: amount of episodes to sample
"""
def compute_variable_plots(path, num_episodes=10, multi_actor=False):
    if not multi_actor:
        run_actor = run_actor_test
    else:
        run_actor = run_multi_actor_test
    run_data = run_actor(path, num_episodes)
    get_correlation_plots(path, run_data)
    get_time_communication(path, run_data)
    communication_hists(path, run_data)
    cumulative_eaten(path, run_data)
    get_comm_resource_regression(path, run_data)

def get_comm_resource_regression(path, run_data):
    (states, actions, eaten, rewards, metadata) = run_data
    # Get the communication strength and resource collection
    c_means = get_comm_strength(actions).reshape(actions.shape[0], -1).mean(axis=1).flatten()
    eaten = eaten[:,-1].flatten()[:, np.newaxis]
    # Perform linear regression
    res = OLS(c_means, np.hstack((eaten, np.ones_like(eaten)))).fit()
    fig = plt.figure()
    plt.title("Communication strength vs resource collection")
    plt.xlabel("Collected resources")
    plt.ylabel("Mean(Communication strength)")
    plt.scatter(eaten, c_means, c='b')
    plt.plot(eaten, res.fittedvalues, label="Fitted line", color='r', linestyle='--')
    plt.plot([], [], ' ', label="$r^2$ = {:.2f}".format(res.rsquared))
    plt.plot([], [], ' ', label="y = {:.2f}x + {:.2f}".format(res.params[0], res.params[1]))
    plt.plot([], [], ' ', label="p-value: {:.2f}".format(res.pvalues[0]))
    plt.plot([], [], ' ', label="t-value: {:.2f}".format(res.tvalues[0]))
    plt.legend()
    plt.grid()
    fig.savefig(os.path.join(path, "comm_resource_regression.png"))

"""
Creates scatterplots of two variables. The variables of interest are:
(a_comm, a_att), (c_strength, d_to_patch), (c_strength, internal_e).

Parameters:
- path: path to save plots to
- run_data: data from running the actor network on the environment with run_actor_test()

Returns: None
"""
def get_correlation_plots(path, run_data):
    (states, actions, eaten, rewards, metadata) = run_data
    a, c, energy, c_strength, d_to_patch = get_derived_data(states, actions)
    # Plot correlation between communication and attention
    #fig1 = plt.figure()
    ax = jointplot(x=c.flatten(), y=a.flatten(), kind='hex', marginal_kws={'bins': 30})
    plt.suptitle("Relation between communication and attention")
    plt.xlabel("Communication")
    plt.ylabel("Attention")
    ax.savefig(os.path.join(path, "comm_att_relation.png"))
    # Plot correlation between communication strength and distance to patch
    ax = jointplot(x=c_strength.flatten(), y=d_to_patch.flatten(), kind='hex', marginal_kws={'bins': 30})
    plt.suptitle("Relation between communication strength and distance to patch")
    plt.xlabel("Communication strength")
    plt.ylabel("Distance to patch")
    ax.savefig(os.path.join(path, "comm_dist_relation.png"))
    # Plot correlation between communication strength and internal energy
    ax = jointplot(x=c_strength.flatten(), y=energy.flatten(), kind="hex", marginal_kws={'bins': 30})
    plt.suptitle("Relation between communication strength and internal energy")
    plt.xlabel("Communication strength")
    plt.ylabel("Internal energy")
    ax.savefig(os.path.join(path, "comm_energy_relation.png"))

def cumulative_eaten(path, run_data): 
    (states, actions, eaten, rewards, metadata) = run_data
    # Make plot of the removed metabolism
    fig, ax = plt.subplots(figsize=(10,5))
    ax.set_title("Resources eaten by agents")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Cumulative resources eaten")
    ax.plot(eaten.mean(axis=0), color='b')
    fig.savefig(os.path.join(path, "resources_eaten.png"))



def communication_hists(path, run_data):
    (states, actions, eaten, rewards, metadata) = run_data
    a, c, energy, c_strength, d_to_patch = get_derived_data(states, actions)
    in_patch = np.where(d_to_patch <= 0, True, False)
    ip_counts = np.unique(in_patch.sum(axis=2), return_counts=True)[1]
    energy = energy.flatten()
    d_to_patch = d_to_patch.flatten()
    c_strength = c_strength.flatten()
    e_c, e_bins = np.histogram(energy, bins=40, range=(energy.min(), energy.max()), weights=c_strength)
    d_c, d_bins = np.histogram(d_to_patch, bins=40, range=(d_to_patch.min(), d_to_patch.max()), weights=c_strength)

    fig, ax = plt.subplots(1,3, figsize=(15,4))
    ax[0].set_title("Distribution of energy when communicating")
    ax[0].set_xlabel("Energy")
    ax[0].set_ylabel("p(communication)")
    bin_widths = np.diff(e_bins)
    ax[0].bar(e_bins[:-1], e_c, width=bin_widths, align='edge', edgecolor='black')

    ax[1].set_title("Distance to patch when communicating")
    ax[1].set_xlabel("Distance to patch")
    ax[1].set_ylabel("p(communication)")
    bin_widths = np.diff(d_bins)
    ax[1].bar(d_bins[:-1], d_c, width=bin_widths, align='edge', edgecolor='black')

    ax[2].set_title("Categories of resource collection when communicating")
    categories = ["None", "One", "Both"]
    ax[2].bar(categories[:len(ip_counts)], ip_counts, edgecolor='black')
    ax[2].set_ylabel("p(communication)")
    ax[2].set_xlabel("Number of agents inside patch")
    fig.savefig(os.path.join(path, "comm_hists.png"))

    fig2, ax = plt.subplots(2,2, figsize=(10,8))
    ax[0,0].set_title("Observed values for $c(t)$")
    ax[0,0].set_xlabel("Values")
    ax[0,0].set_ylabel("Density")
    ax[0,0].hist(c.flatten(), bins=30, density=True)
    ax[1,0].set_title("Observed values for $a(t)$")
    ax[1,0].set_xlabel("Values")
    ax[1,0].set_ylabel("Density")
    ax[1,0].hist(a.flatten(), bins=30, density=True)
    ax[0,1].set_title("Observed values for $c(t)$ if $a(t) > 0.7$")
    ax[0,1].set_xlabel("Values")
    ax[0,1].set_ylabel("Density")
    ax[0,1].hist(c[a > 0.7].flatten(), bins=30, density=True)
    ax[1,1].set_title("Observed values for $a(t)$ if $c(t) > 0.7$")
    ax[1,1].set_xlabel("Values")
    ax[1,1].set_ylabel("Density")
    ax[1,1].hist(a[c > 0.7].flatten(), bins=30, density=True)
    plt.tight_layout()
    fig2.savefig(os.path.join(path, "comm_channel_hists.png"))


"""
Seperate the run data into time bins and seeing 
how communication is distributed temporally over an episode.
For this a 6-bin histogram is generated

Parameters:
- path: path to save the plot to
- run_data: data gathered from running the final actor network
"""
def get_time_communication(path, run_data):
    (states, actions, eaten, rewards, metadata) = run_data
    c_strength = get_comm_strength(actions)
    c_strength_bins = np.split(c_strength, 12, axis=1)
    c_amount = [bin.sum() for bin in c_strength_bins]
    x_range = np.arange(1,13)
    fig = plt.figure()
    plt.bar(x_range, c_amount)
    plt.title("Density of communication throughout an average episode")
    plt.xlabel("Bin")
    plt.ylabel("Average communication strength")
    fig.savefig(os.path.join(path,"temporal_communication.png"))

def get_learned_comm_hist(path):
    actions = get_data(path, "actions.npy")
    c_strength = get_comm_strength(actions)
    c_strength = np.reshape(c_strength, (6, -1))
    fig = plt.figure()
    plt.title("Emergence of communication")
    for i in range(c_strength.shape[0]):
        plt.hist(c_strength[i], label=f"Bin {i}.", density=True, bins=50, range=(c_strength.min(), c_strength.max()), alpha=0.3)
    plt.xlabel("Communication strength")
    plt.ylabel("Density")
    plt.legend()
    fig.savefig(os.path.join(path, "emergence_comm.png"))

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
    return_data = get_grouped_data(path, "returns.npy").mean(axis=3).sum(axis=2)
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

def get_comm_filter(actions):
    communication = actions[:,:,:,2]
    attention_other = np.flip(actions[:,:,:,3],axis=2)
    comm_filter = np.sqrt(communication*attention_other)
    return comm_filter > .7

"""
This function derives the data needed for RQ1 plots, based on the stored data
"""
def rq1_data(patch_info, agents_state, actions):
    patch_position = patch_info[0][:2]
    energy = agents_state[:,:,:,4]
    positions = agents_state[:,:,:,:2]
    dist_agents = np.sqrt(np.sum(np.power(np.diff(positions, axis=2),2),axis=3))[:,:,0]
    # We compute the distance of each agent w.r.t the edge of the patch
    radius = patch_info[0][2]
    dist_agents_patch = np.sqrt(np.sum(np.power(positions - patch_position[np.newaxis,np.newaxis,np.newaxis,:],2),axis=3)) - radius
    if actions.shape[3] > 2:
        comm_filter = get_comm_strength(actions)
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
    fig1.savefig(os.path.join(path, "comm_steps.png"))

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
    plt.colorbar(sm, ax=ax, label="Communication strength")
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
    plt.colorbar(sm, ax=ax, label="Communication strength")
    plt.legend()
    return fig

def plot_color_lines(x,y,c,ax,cmap=plt.get_cmap('viridis')):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # dummy array — colorbar only needs cmap + norm
    lc = LineCollection(segments, cmap=cmap, linewidth=2)
    c = c.astype(float)
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
        comms_state = np.reshape(actions[:,:,:,-2], (-1, n_agents))
        agents_data.append(("Comms", comms_state))
    if action_dim >= 4:
        attention_state = np.reshape(actions[:,:,:,-1], (-1, n_agents))
        agents_data.append(("Attention", attention_state))
    agents_data.append(("Energy", energy_state))
    return resource_state, agents_data

def plot_env_vars(resource_state, agents_data, axes, trail=100, a_colors=COLORS):
    font_size = 11
    def scale_y(ax, name, data):
        ax.set_ylim([data.min()-data.max()*0.1, data.max()*1.1])
        if name in ["Attention", "Comms"]:
            ax.set_ylim([-0.1, 1.1])
    resource = axes[0].plot(resource_state[0], c='g')[0]
    axes[0].set_ylabel("Patch", fontsize=font_size)
    axes[0].set_xlim([0,trail])
    scale_y(axes[0], "Resource", resource_state)
    axes[0].tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
    agent_plots = []
    for i,(name,d) in enumerate(agents_data):
        agent_plots.append([axes[i+1].plot(d[0, i_a], c=a_colors[i_a])[0] for i_a in range(d.shape[-1])])
        scale_y(axes[i+1], name, d)
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

def plot_final_welfare(path, states, metadata):
    reward = energy_to_reward(states[:,:,:,4], metadata["e_max"])
    nsw = compute_NSW(reward, axis=2).mean(axis=1)
    fig = plt.figure()
    plt.plot(nsw)
    plt.title("Welfare of agents across episodes")
    plt.xlabel("Episode")
    plt.ylabel("$mean(NSW(r_{ind,1},r_{ind,2}))$")
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
    labels = []
    for i,path in enumerate(paths):
        label = os.path.basename(os.path.dirname(path))
        labels.append(label)
        plt.hist(returns[i], bins=max(num_episodes//4,40), range=(returns.min(), returns.max()), label=label, alpha=0.4)  
    for i in range(returns.shape[0]):
        for j in range(returns.shape[0]):
            if j > i:
                plt.plot([], [], ' ', label=f"({labels[i]},{labels[j]}): p={round(ttest_ind(returns[i], returns[j]).pvalue,3)}")
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
    steps = data.shape
    fig = plt.figure()
    if name == "critic":
        data = np.clip(data, 0, 100)
    plt.title(f"{name} loss of agents")
    plt.xlabel("Timestep")
    plt.ylabel(f"{name} loss")
    plt.plot(data, color=colors[0])
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
    passttt