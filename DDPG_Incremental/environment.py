import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os
from abc import ABC, abstractmethod
from functools import partial

# Helper function for computing the Nash Social Welfare function (aka geometric mean)
def compute_NSW(rewards):
    #print("Rewards: ", rewards)
    NSW = np.power(np.prod(rewards), 1/rewards.shape[0])
    #print("Welfare: ", NSW.item())
    return NSW
    
class NAgentsEnv():
    def __init__(self, patch_radius=10,s_init=10, e_init=5, eta=0.05, beta=0.05, alpha=2, gamma=0, step_max=400, x_max=50, y_max=50, v_max=4, n_agents=2, obs_others=False, obs_range=80, in_patch_only=False, p_welfare=0, rof=0, patch_resize=False, **kwargs):
        self.x_max = x_max
        self.y_max = y_max
        self.v_max = v_max
        # Eta and gamma are used for computing the rendering (don't remove them!!)
        self.eta = eta
        self.gamma = gamma
        self.step_max = step_max
        self.step_idx = 0
        self.rof = rof
        self.patch = Patch(x_max/2, y_max/2, patch_radius, s_init, eta=eta, gamma=gamma, rof=rof, patch_resize=patch_resize)
        self.agents = [Agent(0,0,x_max,y_max,e_init,v_max, alpha=alpha, beta=beta, id=i) for i in range(n_agents)]
        self.n_agents = n_agents
        self.alpha = alpha
        self.beta = beta
        self.e_init = e_init
        self.obs_others = obs_others
        self.obs_range = obs_range
        self.in_patch_only = in_patch_only
        self.patch_resize = patch_resize
        self.p_welfare = p_welfare
        # Initialize latest observation states
        self.latest_obs = np.zeros((self.n_agents, self.n_agents, self.agents[0].num_vars-1))
        self.latest_patch = 0

    def get_action_space(self):
        action_dim = 2 # We use two acceleration terms (one for x and one for y)
        action_range = [-self.v_max, self.v_max]
        return action_dim, action_range
    
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
            "obs_range": self.obs_range,
            "in_patch_only": self.in_patch_only,
            "p_welfare": self.p_welfare,
            "rof":self.rof,
            "patch_resize":self.patch_resize
        }

    def size(self):
        return self.x_max, self.y_max

    """
    For the state generator version 2 below, I will make a few major changes.
    The patch resource state will now only be seen if the agent is in the patch, or if communication is enabled (obs_others) and a close-by agent is in the patch, and if in_patch_only is false, it will always show the resource state.
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
            agents_obs[i, ptr:ptr+self.patch.num_vars-1] = patch_state[:-1]
            ptr += self.patch.num_vars-1
            # Add patch resource info to state
            if (np.any(is_nearby[i] & in_patch) and self.obs_others) or in_patch[i] or not self.in_patch_only:
                self.latest_patch = patch_state[-1]
            agents_obs[i, ptr] = self.latest_patch
            ptr += 1
            # Add position and velocity information of nearby agents (including self) to state
            if self.obs_others:
                for j in range(self.n_agents):
                    if is_nearby[i,j] and i != j:
                        self.latest_obs[i,j] = agents_state[j][:4] # We only provide the agent with the latest observed information  
                    if i != j:
                        agents_obs[i, ptr:ptr+4] = self.latest_obs[i,j]
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
        rng = np.random.default_rng(seed=seed)
        # Initialize rng_key and state arrays
        agents_state = np.zeros((self.n_agents, self.agents[0].num_vars))
        # Reset the patch
        patch_state = self.patch.reset()
        # Generate random position for each agent
        for i in range(self.n_agents):
            x = rng.random()*self.x_max
            y = rng.random()*self.y_max
            pos = [x,y]
            agents_state[i] = self.agents[i].reset(*pos, rng)
        # Reset counter
        step_idx = 0
        # Store agents' observations
        agents_obs = self.get_states(agents_state, patch_state)
        # Return the states and observations
        env_state = (agents_state, patch_state, step_idx) 
        return env_state, agents_obs

    def step(self, env_state, *actions):
        (agents_state, patch_state, step_idx) = env_state
        rewards = np.empty(self.n_agents)
        welfare = 0
        penalties = np.empty((self.n_agents, 1))
        is_in_patch = np.empty(self.n_agents)
        tot_eaten = 0
        for i,action in enumerate(actions):
            # Flatten array if needed
            action = action.flatten()
            # Update agent position
            a_state = agents_state[i]
            agents_state[i, :agents_state.shape[1]] = self.agents[i].update_position(a_state, action)
            # Update agent energy
            agent_state, reward, s_eaten, penalty = self.agents[i].update_energy(agents_state, patch_state, action, dt=0.1)
            is_in_patch[i] = s_eaten != 0
            penalties[i,:] = penalty
            # Add agent reward to reward vector
            rewards[i] = reward
            tot_eaten += s_eaten
            agents_state[i] = agent_state
        # Compute welfare
        welfare = compute_NSW(rewards) 
        # Compute reward with welfare
        rewards = (1-self.p_welfare)*rewards + self.p_welfare*welfare
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
    def __init__(self,x,y,x_max,y_max,e_init,v_max,alpha=0.4, beta=0.25, id=0):
        # Hyperparameters of dynamical system
        self.alpha = alpha # penalizing coëfficient for movement
        self.beta = beta # amount of eating per timestep
        # General variables
        self.v_max = v_max
        self.size = [x_max, y_max]
        self.e_init = e_init
        self.id = id
        self.max_penalty = np.linalg.norm(np.array([self.v_max]*2))
        self.num_vars = 5 # Variables of interest: (x,y,v_x,v_y,e)
    
    def reset(self,x,y,rng):
        agent_state = np.zeros(self.num_vars)
        agent_state[:2] = np.array([x,y])
        vs = 2*self.v_max*rng.random(size=2) - self.v_max
        agent_state[2:4] = vs
        agent_state[4] = self.e_init
        return agent_state

    def dist_to_patch(self,agent_state, patch_state):
        diff = agent_state[:2] - patch_state[:2]
        return np.linalg.norm(diff, 2)
    
    def is_in_patch(self,agent_state,patch_state):
        dist = self.dist_to_patch(agent_state, patch_state)
        return dist <= patch_state[3]

    def is_in_rof(self,agent_state,patch_state):
        dist = self.dist_to_patch(agent_state, patch_state)
        return dist > patch_state[3] and dist <= (patch_state[3] + patch_state[2])
        

    def update_energy(self,agents_state,patch_state,action,dt=0.1):
        # Amount eaten depends on other agents in patch
        agent_state = agents_state[self.id]
        who_in = [self.is_in_patch(agents_state[i], patch_state) for i in range(len(agents_state))]
        s_eaten = who_in[self.id]*self.beta*patch_state[3]
        if np.any(who_in):
            s_eaten /= np.sum(who_in)
        # Penalty terms for the environment (inverted for minimization instead of maximization) 
        p_still = 0.2
        action_penalty = ((1-p_still)*(np.linalg.norm(action.at[:2].get())/self.max_penalty)+p_still)*self.alpha
        rof_penalty = (self.is_in_rof(agent_state,patch_state)).astype(int)*self.alpha
        # Update step (differential equation)
        de = s_eaten - dt*(action_penalty + rof_penalty)
        # If agent has negative or zero energy, put the energy value at zero and consider the agent dead
        # Also agent can't have more energy than it's initial energy value
        #print("Start: ", agent_state[-1])
        agent_state[-1] = np.clip(agent_state[-1]+de, 0, self.e_init)
        #print("End: ", agent_state[-1])
        reward = np.clip(agent_state[-1]/self.e_init, 0, 1)
        penalties = action_penalty
        return agent_state, reward, s_eaten, penalties
        
    def update_position(self,agent_state,action, dt=0.1):
        # Functions needed to bound the allowed actions
        v_bounded = lambda v: np.clip(v, -self.v_max, self.v_max)
        # Compute action values
        pos = agent_state[:2].copy()
        vel = agent_state[2:4].copy()
        acc = action.at[:2].get()
        damp = 0.3
        # Update position
        pos += dt*vel 
        pos = np.mod(pos, self.size)
        # Update velocity
        vel = v_bounded(vel + dt*(acc-damp*vel))
        agent_state[:2] = pos
        agent_state[2:4] = vel
        return agent_state

class Patch:
    def __init__(self, x,y,radius,s_init, eta=0.1, gamma=0.01, rof=0, patch_resize=False):
        # Hyperparameters of dynamical system
        self.eta = eta # regeneration rate of resources
        self.gamma = gamma # decay rate of resources
        # General variables
        self.pos = np.array([x,y])
        self.rof = rof
        self.radius = radius
        self.patch_resize = patch_resize
        self.s_init = s_init
        self.num_vars = 5 # Variables of interest: (x,y,r,rof,s)
    def get_radius(self):
        return self.radius
    def update_resources(self, patch_state, eaten, dt=0.1):
        resources = patch_state[4]
        
        # Lotka volterra dynamics
        #scalars = np.array([dt*self.eta, -dt*self.gamma, -1])
        #values = np.array([resources,np.power(resources,2),eaten])
        #ds = np.dot(scalars.T, values)
        
        # Linear dynamics
        ds = dt*self.eta - eaten
        patch_state[4] = np.clip(resources + ds, 0, self.s_init) 
        if self.patch_resize:
            patch_state[3] = self.radius*0.5 + 0.5*self.radius*(patch_state[4]/self.s_init)
        return patch_state
    def reset(self):
        patch_state = np.zeros(self.num_vars)
        patch_state[:2] = self.pos
        patch_state[2] = self.rof
        patch_state[3] = self.radius
        patch_state[4] = self.s_init
        return patch_state