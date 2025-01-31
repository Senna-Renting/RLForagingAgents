import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os
from abc import ABC, abstractmethod
from functools import partial

# TODO: create an abstract class for the environment from which the specific experiments then can be build
class Environment(ABC):
    def __init__(self, patch_radius=2, s_init=10, e_init=1, eta=0.1, beta=0.5, alpha=0.1, gamma=0.01, step_max=400, x_max=5, y_max=5, v_max=0.1):
        self.x_max = x_max
        self.y_max = y_max
        self.v_max = v_max
        # Eta and gamma are used for computing the rendering (don't remove them!!)
        self.eta = eta
        self.gamma = gamma
        self.step_max = step_max
        self.step_idx = 0
        self.patch = Patch(x_max/2, y_max/2, patch_radius, s_init, eta=eta, gamma=gamma)

    def get_action_space(self):
        action_dim = 2 # We use two acceleration terms (one for x and one for y)
        action_range = [-2*self.v_max, 2*self.v_max]
        return action_dim, action_range

    @abstractmethod
    def step(self, action):
        ...
    @abstractmethod
    def reset(self, seed=0):
        ...
    @abstractmethod
    def get_state_space(self):
        ...

# TODO: make a two (or N-) agent version of the single-patch foraging environment
class NAgentsEnv(Environment):
    def __init__(self, patch_radius=0.5, s_init=10, e_init=1, eta=0.1, beta=0.5, alpha=0.0025, gamma=0.01, step_max=400, x_max=5, y_max=5, v_max=0.1, n_agents=2, obs_others=False, obs_range=8):
        super().__init__(patch_radius=patch_radius, s_init=s_init, e_init=e_init, eta=eta, beta=beta, alpha=alpha, gamma=gamma, step_max=step_max, x_max=x_max, y_max=y_max, v_max=v_max)
        beta = beta / n_agents # This adjustment is done to keep the resource dynamics similar across different agent amounts
        self.agents = [Agent(0,0,x_max,y_max,e_init,v_max, alpha=alpha, beta=beta) for i in range(n_agents)]
        self.n_agents = n_agents
        self.alpha = alpha
        self.beta = beta
        self.e_init = e_init
        self.obs_others = obs_others
        self.obs_range = obs_range

    def get_params(self):
        return {
            "n_agents": self.n_agents, 
            "x_max": self.x_max,
            "y_max": self.y_max,
            "v_max": self.v_max,
            "patch_radius": self.patch.get_radius(),
            "s_init": self.patch.s_init,
            "e_init": self.e_init,
            "eta": self.eta,
            "beta": self.beta,
            "alpha": self.alpha,
            "env_gamma": self.gamma,
            "step_max": self.step_max,
            "obs_others": self.obs_others,
            "obs_range": self.obs_range
        }

    def size(self):
        return self.x_max, self.y_max

    # TODO: maybe implement a toggle for viewing the energy state of other agents
    def get_states_v1(self, agents_state, patch_state):
        energy_toggle = int(self.n_agents>1)*1
        obs_size = self.n_agents*self.agents[0].num_vars +(self.n_agents-1)*self.comm_dim + self.patch.num_vars - energy_toggle
        if not self.obs_others:
            obs_size -= (self.n_agents-1)*self.agents[0].num_vars - energy_toggle
        agents_obs = np.zeros((self.n_agents, obs_size))
        state_without_energy = agents_state[:,:agents_state.shape[1]-self.comm_dim-1]
        comm_vec = agents_state[:,agents_state.shape[1]-self.comm_dim:]
        energy_vec = agents_state[:,4]
        a_state = state_without_energy.flatten()
        for i in range(self.n_agents):
            if not self.obs_others:
                a_state = state_without_energy[i,:]
            # Add communication channels and energy back to the agent's state
            a_comm_state = np.concatenate([a_state, comm_vec[i,:], [energy_vec[i]]])
            obs = np.concatenate([patch_state, a_comm_state])
            # obs = jnp.expand_dims(obs, 0) # May only be necessary for one agent case?
            agents_obs[i,:] = obs
        return agents_obs

    """
    For the state generator version 2 below, I will make a few major changes.
    The patch resource state will now only be seen if the agent is in the patch, or if communication is enabled (obs_others) and a close-by agent is in the patch.
    Agents communicate their velocity and position with the other agents only when nearby (within obs_range) if obs_others is enabled.
    When either the patch resource cannot be seen, or other agents are not nearby enough, we will assume zero values for their states, as this should remove dependence on those state values for the neural networks.
    """
    def get_states(self, agents_state, patch_state):
        # Initialize size of state
        obs_size = self.get_state_space()
        is_nearby = self.get_nearby_agents(agents_state)
        in_patch = np.array([self.agents[i].is_in_patch(agents_state[i], patch_state) for i in range(self.n_agents)])
        agents_obs = np.zeros((self.n_agents, obs_size))
        # Compute components of state
        for i in range(self.n_agents):
            ptr = 0
            agents_obs[i, ptr:self.agents[i].num_vars] = agents_state[i]
            ptr += self.agents[i].num_vars
            agents_obs[i, ptr:ptr+self.patch.num_vars-1] = patch_state[:3]
            ptr += self.patch.num_vars-1
            # Add patch resource info to state
            if (np.any(is_nearby[i] & in_patch) and self.obs_others) or in_patch[i]:
                agents_obs[i, ptr] = patch_state[3]
            ptr += 1
            # Add position and velocity information of nearby agents (including self) to state
            if self.obs_others:
                for j in range(self.n_agents):
                    if is_nearby[i,j] and i != j:
                        agents_obs[i, ptr:ptr+4] = agents_state[j][:4]
                    if i != j:
                        ptr += 4
        return agents_obs

    def get_nearby_agents(self, agents_state):
        nearby_mat = np.empty((self.n_agents, self.n_agents), dtype='bool')
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                x_diff = agents_state[i][0] - agents_state[j][0]
                y_diff = agents_state[i][1] - agents_state[j][1]
                dist = np.sqrt(x_diff**2 + y_diff**2)
                nearby_mat[i,j] = dist < self.obs_range
        return nearby_mat

    def get_state_space(self):
        obs_size = self.agents[0].num_vars + self.patch.num_vars
        if self.obs_others:
            obs_size += 4*(self.n_agents-1) # Add indices for tracking position and velocity of other agents when nearby
        return obs_size 
    
    def reset(self, seed=0):
        # Initialize rng_key and state arrays
        a_keys = jax.random.split(jax.random.PRNGKey(seed), self.n_agents)
        agents_state = np.zeros((self.n_agents, self.agents[0].num_vars))
        # Reset the patch
        patch_state = self.patch.reset()
        # Generate random position for each agent
        for i in range(self.n_agents):
            def is_in_patch(a):
                (pos, _) = a
                x_diff = pos[0] - patch_state[0]
                y_diff = pos[1] - patch_state[1]
                dist = jnp.sqrt(x_diff**2 + y_diff**2)
                return dist <= patch_state[2]
            def get_coordinates(a):
                (_, key) = a
                subkey, key = jax.random.split(key)
                x = jax.random.uniform(subkey, minval=0,maxval=self.x_max)
                y = jax.random.uniform(key, minval=0,maxval=self.y_max)
                return ([x,y], key)
            (pos,key) = jax.lax.while_loop(is_in_patch, get_coordinates, get_coordinates((0,a_keys[i])))
            agents_state[i] = self.agents[i].reset(*pos)
        # Reset counter
        step_idx = 0
        # Store agents' observations
        agents_obs = self.get_states(agents_state, patch_state)
        # Return the states and observations
        env_state = (agents_state, patch_state, step_idx) 
        return env_state, agents_obs

    def step(self, env_state, *actions):
        (agents_state, patch_state, step_idx) = env_state
        rewards = np.empty((self.n_agents, 1))
        n_penalties = 1
        penalties = np.empty((self.n_agents, n_penalties))
        is_in_patch = np.empty(self.n_agents)
        tot_eaten = 0
        for i,action in enumerate(actions):
            # Flatten array if needed
            action = action.flatten()
            # Update agent position
            a_state = agents_state[i]
            agents_state[i, :agents_state.shape[1]] = self.agents[i].update_position(a_state, action)
            # Update agent energy
            agent_state, reward, s_eaten, penalty = self.agents[i].update_energy(a_state, patch_state, action, dt=0.1)
            is_in_patch[i] = s_eaten != 0
            penalties[i,:] = penalty
            # Add agent reward to reward vector
            rewards[i,:] = reward
            tot_eaten += s_eaten
            agents_state[i] = agent_state
        # Update patch resources
        patch_state = self.patch.update_resources(patch_state, tot_eaten, dt=0.1)
        # Update counter
        step_idx += 1
        # Update states AFTER dynamical system updates of each agent have been made
        next_states = self.get_states(agents_state, patch_state)
        # When any of the agents dies, the environment is terminated 
        terminated = False #np.any(agents_state[:,-1] == 0)
        truncated = step_idx >= self.step_max
        env_state = (agents_state, patch_state, step_idx)
        agents_info = (penalties, is_in_patch)
        return env_state, next_states, (rewards, agents_info), terminated, truncated, None # None is to have similar output shape as gym API



class Agent:
    def __init__(self,x,y,x_max,y_max,e_init,v_max,alpha=0.4, beta=0.25):
        # Hyperparameters of dynamical system
        self.alpha = alpha # penalizing coëfficient for movement
        self.beta = beta # amount of eating per timestep
        # General variables
        self.v_max = v_max
        self.x_max = x_max
        self.y_max = y_max
        self.e_init = e_init
        self.num_vars = 5 # Variables of interest: (x,y,v_x,v_y,e)
    
    def reset(self,x,y,seed=0):
        v_key, theta_key = jax.random.split(jax.random.PRNGKey(seed))
        agent_state = np.zeros(self.num_vars)
        agent_state[:2] = np.array([x,y])
        x_dot = jax.random.uniform(v_key, minval=-self.v_max, maxval=self.v_max)
        y_dot = jax.random.uniform(v_key, minval=-self.v_max, maxval=self.v_max)
        agent_state[2:4] = np.array([x_dot, y_dot])
        agent_state[4] = self.e_init
        return agent_state
    
    def is_in_patch(self,agent_state,patch_state):
        x_diff = agent_state[0] - patch_state[0]
        y_diff = agent_state[1] - patch_state[1]
        dist = np.sqrt(x_diff**2 + y_diff**2)
        return dist <= patch_state[2]

    def update_energy(self,agent_state,patch_state,action,dt=0.1):
        s_eaten = (self.is_in_patch(agent_state,patch_state)).astype(int)*self.beta*patch_state[3]
        # Penalty terms for the environment (inverted for minimization instead of maximization)
        max_penalty = np.linalg.norm(np.array([self.v_max]*2)) 
        action_penalty = (np.linalg.norm(action.at[:2].get())/max_penalty)*self.alpha
        # Update step (differential equation)
        de = dt*(s_eaten - action_penalty)
        reward = de + 1*self.alpha # Ensure positive reward for algorithm
        # If agent has negative or zero energy, put the energy value at zero and consider the agent dead
        agent_state[-1] = np.max([0., agent_state[-1] + de])
        penalties = action_penalty
        return agent_state, reward, s_eaten, penalties
        
    def update_position(self,agent_state,action):
        # Functions needed to bound the allowed actions
        v_bounded = lambda v: max(min(v, self.v_max), -self.v_max)
        # Compute action values
        x_dot = agent_state[2]
        y_dot = agent_state[3]
        x_dot = v_bounded(x_dot + action.at[0].get())
        y_dot = v_bounded(y_dot + action.at[1].get())
        agent_state[2:4] = [x_dot,y_dot]
        # Update position
        x = (agent_state[0] + agent_state[2]) % self.x_max
        y = (agent_state[1] + agent_state[3]) % self.y_max
        agent_state[:2] = [x,y]
        return agent_state

class Patch:
    def __init__(self, x,y,radius,s_init, eta=0.1, gamma=0.01):
        # Hyperparameters of dynamical system
        self.eta = eta # regeneration rate of resources
        self.gamma = gamma # decay rate of resources
        # General variables
        self.pos = np.array([x,y])
        self.radius = radius
        self.s_init = s_init
        self.num_vars = 4 # Variables of interest: (x,y,r,s)
    def get_radius(self):
        return self.radius
    def update_resources(self, patch_state, eaten, dt=0.1):
        resources = patch_state[3]
        s_growth = np.multiply(self.eta,resources)
        s_decay = np.multiply(self.gamma,np.power(resources,2))
        ds = dt*(s_growth - s_decay - eaten)
        patch_state[3] = np.array(max(0,resources + ds), dtype=np.float32)
        return patch_state
    def reset(self):
        patch_state = np.zeros(self.num_vars)
        patch_state[:2] = self.pos
        patch_state[2] = self.radius
        patch_state[3] = self.s_init
        return patch_state