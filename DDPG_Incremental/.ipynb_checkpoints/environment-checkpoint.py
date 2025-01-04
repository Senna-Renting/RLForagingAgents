import jax
import jax.numpy as jnp
import pygame
from pygame.locals import *
import numpy as np
import matplotlib.pyplot as plt
from pygame_recorder import PygameRecord
import os
from abc import ABC, abstractmethod
from functools import partial

# TODO: create an abstract class for the environment from which the specific experiments then can be build
class Environment(ABC):
    def __init__(self, seed=0, patch_radius=2, s_init=10, e_init=1, eta=0.1, beta=0.5, alpha=0.1, gamma=0.01, step_max=400, x_max=5, y_max=5, v_max=0.1):
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
    def __init__(self, seed=0, patch_radius=0.5, s_init=10, e_init=1, eta=0.1, beta=0.5, alpha=0.025, gamma=0.01, step_max=400, x_max=5, y_max=5, v_max=0.1, n_agents=2, sw_fun=lambda x:0, obs_others=False, reward_dim=1, comm_dim=0):
        super().__init__(seed=seed, patch_radius=patch_radius, s_init=s_init, e_init=e_init, eta=eta, beta=beta, alpha=alpha, gamma=gamma, step_max=step_max, x_max=x_max, y_max=y_max, v_max=v_max)
        beta = beta / n_agents # This adjustment is done to keep the resource dynamics similar across different agent amounts
        self.agents = [Agent(0,0,x_max,y_max,e_init,v_max, alpha=alpha, beta=beta) for i in range(n_agents)]
        self.n_agents = n_agents
        self.sw_fun = sw_fun
        self.alpha = alpha
        self.e_init = e_init
        self.obs_others = obs_others
        self.comm_dim = comm_dim
        self.penalty_dim = 1+int(comm_dim>0)
        self.reward_dim = self.penalty_dim+reward_dim # 1: action_norm, int(comm_dim>0): comm_norm 

    def size(self):
        return self.x_max, self.y_max

    # TODO: maybe implement a toggle for viewing the energy state of other agents
    def get_states(self, agents_state, patch_state):
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

    def get_state_space(self):
        env_state, agents_obs = self.reset()
        return agents_obs.shape[1]
        
    # TODO: Convert tried jax jit code back to regular numpy code
    def reset(self, seed=0):
        # Initialize rng_key and state arrays
        a_keys = jax.random.split(jax.random.PRNGKey(seed), self.n_agents)
        agents_state = np.zeros((self.n_agents, self.agents[0].num_vars+(self.n_agents-1)*self.comm_dim))
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
            agent_state = self.agents[i].reset(*pos)
            # Concatenate zero communication vector with each agent's calculated coordinates
            agents_state[i,:] = jnp.concatenate([agent_state,jnp.zeros((self.n_agents-1)*self.comm_dim)])
        # Reset counter
        step_idx = 0
        # Store agents' observations
        agents_obs = self.get_states(agents_state, patch_state)
        # Return the states and observations
        env_state = (agents_state, patch_state, step_idx) 
        return env_state, agents_obs

    def step(self, env_state, *actions):
        (agents_state, patch_state, step_idx) = env_state
        rewards = np.empty((self.n_agents, self.reward_dim))
        tot_eaten = 0
        for i,action in enumerate(actions):
            # Flatten array if needed
            action = action.flatten()
            # Obtain communication vector
            comm_vec = jnp.array([action[action.shape[0]-self.comm_dim:] for j,action in enumerate(actions) if i!=j]).flatten()
            # Update agent position
            a_state = agents_state[i, :agents_state.shape[1]-comm_vec.shape[0]]
            agents_state[i, :agents_state.shape[1]-comm_vec.shape[0]] = self.agents[i].update_position(a_state, action)
            agents_state[i, agents_state.shape[1]-comm_vec.shape[0]:] = comm_vec
            # Update agent energy
            agent_state, reward, s_eaten = self.agents[i].update_energy(a_state, patch_state, action, dt=0.1)
            # Add agent reward to reward vector
            if (self.reward_dim-self.penalty_dim) == 2:
                rewards[i,:-1] = reward
            else:
                rewards[i,:] = reward
            tot_eaten += s_eaten
            agents_state[i, :agents_state.shape[1]-comm_vec.shape[0]] = agent_state
        # Update patch resources
        patch_state = self.patch.update_resources(patch_state, tot_eaten, dt=0.1)
        # Apply social welfare function here (for now only depends on amount eaten by agents together)
        social_welfare = self.sw_fun(np.sum(rewards[:,:-1], axis=1))
        if (self.reward_dim-self.penalty_dim) == 2:
            rewards[:,-1] = np.full(self.n_agents, social_welfare)
        # Update counter
        step_idx += 1
        # Update states AFTER dynamical system updates of each agent have been made
        next_states = self.get_states(agents_state, patch_state)
        # When any of the agents dies, the environment is terminated 
        terminated = np.any(agents_state[:,-1] == 0)
        truncated = step_idx >= self.step_max
        #print(agents_state[:,agents_state.shape[1]-self.comm_dim:])
        env_state = (agents_state, patch_state, step_idx)
        return env_state, next_states, (rewards, social_welfare), terminated, truncated, None # None is to have similar output shape as gym API



class Agent:
    def __init__(self,x,y,x_max,y_max,e_init,v_max, alpha=0.4, beta=0.25):
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
        agent_state[-1] = np.array(self.e_init, dtype=np.float32)
        return agent_state
    
    def is_in_patch(self,agent_state,patch_state):
        x_diff = agent_state[0] - patch_state[0]
        y_diff = agent_state[1] - patch_state[1]
        dist = np.sqrt(x_diff**2 + y_diff**2)
        return dist <= patch_state[2]

    #TODO: Vectorize the penalties as seperate terms for the critic network (I will make it a multi-objective problem this way)
    def update_energy(self,agent_state,patch_state,action,dt=0.1):
        s_eaten = (self.is_in_patch(agent_state,patch_state)).astype(int)*self.beta*patch_state[3]
        # Penalty terms for the environment (inverted for minimization instead of maximization)
        action_penalty = 1/(1+np.linalg.norm(action.at[:2].get()))/400
        comms_penalty = 1/(1+np.linalg.norm(action.at[2:].get()))/400
        # Update step (differential equation)
        de = dt*(s_eaten)
        # Reward vector shape: (de, -action_norm, -comm_norm), we will use a multi-objective approach 
        reward_vec = np.array([de, action_penalty, comms_penalty])
        if action.shape[0] == 2:
            reward_vec = reward_vec[:-1]
        # If agent has negative or zero energy, put the energy value at zero and consider the agent dead
        agent_state[-1] = np.max([0., agent_state[-1] + de])
        return agent_state, reward_vec, s_eaten # reward and amount of resource eaten respectively
        
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

class RenderNAgentsEnvironment:
    def __init__(self, env):
        self.env = env
        self.closed = False
        # Tracking arrays
        self.n_agents = self.env.n_agents
        self.step_idx = 0
        self.agent_poss = jnp.empty((self.env.step_max, self.env.n_agents, 2))
        self.es_agent = jnp.empty((self.env.step_max, self.env.n_agents))
        self.rewards = jnp.empty((self.env.step_max, self.env.n_agents, self.env.reward_dim))
        self.ss_patch = jnp.empty(self.env.step_max)
    
    def reset(self, seed=0):
        (env_state, agents_obs) = self.env.reset(seed=seed)
        (agents_state, patch_state, step_idx) = env_state
        self.step_idx = step_idx
        self.agent_poss = self.agent_poss.at[step_idx-1].set(agents_state[:,:2])
        self.ss_patch = self.ss_patch.at[step_idx-1].set(patch_state[2])
        return (env_state, agents_obs)
    
    def step(self, *actions):
        (env_state, next_ss, (rewards, social_welfare), terminated, truncated, info) = self.env.step(*actions)
        (agents_state, patch_state, step_idx) = env_state
        self.step_idx = step_idx
        self.agent_poss = self.agent_poss.at[step_idx-1].set(agents_state[:,:2])
        self.ss_patch = self.ss_patch.at[step_idx-1].set(patch_state[-1])
        self.es_agent = self.es_agent.at[step_idx-1].set(agents_state[:,-self.env.comm_dim-1])
        self.rewards = self.rewards.at[step_idx-1].set([reward for reward in rewards])
        return (env_state, next_ss, rewards, terminated, truncated, info)

    def render(self, save=True, path=""):
        FPS = 15
        # Generate GIF filename
        n_agents = self.env.n_agents
        fname = f"{n_agents}_agents_one_patch.gif"
        fname = os.path.join(path, fname)

        # Save the plots for the agent and the patch resources
        plt.figure()
        plt.title("Energy of agent(s)")
        plt.xlabel("Timestep")
        plt.ylabel("Energy")
        [plt.plot(self.es_agent[:,i], label=f"Agent {i+1}") for i in range(self.es_agent.shape[1])]
        if self.n_agents > 1:
            plt.legend()
        plt.savefig(os.path.join(path, "agent_energy.png"))

        plt.figure()
        plt.title("Resources in patch")
        plt.xlabel("Timestep")
        plt.ylabel("Resources")
        plt.plot(self.ss_patch)
        plt.savefig(os.path.join(path, "patch_resource.png"))

        plt.figure()
        plt.title("Rewards collected at timesteps")
        plt.xlabel("Timestep")
        plt.ylabel("Reward value")
        [plt.plot(self.rewards[:,i,0], label=f"Agent {i+1}") for i in range(self.rewards.shape[1])]
        if self.n_agents > 1:
            plt.legend()
        plt.savefig(os.path.join(path, "agent_rewards.png"))
        
        # Render game and simultaneously save it as a gif
        with PygameRecord(fname, FPS) as recorder:
            # Pygame screen init
            pygame.init()
            clock = pygame.time.Clock()
            # Get render info
            size_x, size_y = self.env.size()
            scale = 100
            agent_size = size_x/scale
            screen_x = scale*size_x
            screen_y = scale*size_y
            screen = pygame.display.set_mode((screen_x, screen_y))
            red_color = pygame.Color(255, 0, 0)
            white_color = pygame.Color(255,255,255)
            # Select used segments of tracking arrays
            self.agent_poss = self.agent_poss[:self.step_idx - 1]
            self.ss_patch = self.ss_patch[:self.step_idx - 1]
            self.es_agent = self.es_agent[:self.step_idx - 1]
            self.rewards = self.rewards[:self.step_idx - 1]
            # Render on initialization
            for i in range(self.step_idx - 2):
                for event in pygame.event.get():
                    if event.type == QUIT:
                        return
                clock.tick(FPS)
                # Get render info
                size_x, size_y = self.env.size()
                max_ratio_patch = self.ss_patch[i]/(self.env.eta/self.env.gamma)
                patch_pos = self.env.patch.pos
                patch_radius = self.env.patch.get_radius()
                
                #print("Patch color ratio: ",max_ratio_patch)
                patch_color = pygame.Color(0, max(50,int(max_ratio_patch*255)), 0)
                agents_pos = scale*self.agent_poss.at[i].get()
                patch_pos = [scale*patch_pos[0], scale*patch_pos[1]]
                # Draw information on screen
                screen.fill(white_color)
                pygame.draw.circle(screen, patch_color, patch_pos, scale*patch_radius)
                [pygame.draw.circle(screen, red_color, (agents_pos.at[i_a,0].get().item(),agents_pos.at[i_a,1].get().item()), scale*agent_size) for i_a in range(agents_pos.shape[0])]
                # Update screen
                pygame.display.update()
                recorder.add_frame()
            # Close the screen
            pygame.quit()
            if save:
                recorder.save()
