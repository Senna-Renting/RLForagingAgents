import jax
import jax.numpy as jnp
import pygame
from pygame.locals import *
import numpy as np
import matplotlib.pyplot as plt
from pygame_recorder import PygameRecord
import os
from abc import ABC, abstractmethod
from flax.core import FrozenDict

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
    def __init__(self, seed=0, patch_radius=2, s_init=10, e_init=1, eta=0.1, beta=0.5, alpha=0.1, gamma=0.01, step_max=400, x_max=5, y_max=5, v_max=0.1, n_agents=2, sw_fun=lambda x:0, obs_others=False):
        super().__init__(seed=seed, patch_radius=patch_radius, s_init=s_init, e_init=e_init, eta=eta, beta=beta, alpha=alpha, gamma=gamma, step_max=step_max, x_max=x_max, y_max=y_max, v_max=v_max)
        self.agents = [Agent(0,0,x_max,y_max,e_init,v_max, alpha=alpha, beta=beta) for i in range(n_agents)]
        self.n_agents = n_agents
        self.sw_fun = sw_fun
        self.alpha = alpha
        self.e_init = e_init
        self.obs_others = obs_others

    def size(self):
        return self.x_max, self.y_max

    def get_states(self, agents_state, patch_state):
        obs_size = self.n_agents*self.agents[0].num_vars + self.patch.num_vars
        if not self.obs_others:
            obs_size -= (self.n_agents-1)*self.agents[0].num_vars
        agents_obs = np.zeros((self.n_agents, obs_size))
        a_state = agents_state.flatten()
        for i in range(self.n_agents):
            if not self.obs_others:
                a_state = agents_state[i]
            obs = np.concatenate([a_state, patch_state])
            # obs = jnp.expand_dims(obs, 0) # May only be necessary for one agent case?
            agents_obs[i,:] = obs
        return agents_obs

    def get_state_space(self):
        env_state, agents_obs = self.reset()
        print(agents_obs.shape)
        return agents_obs.shape[1]
        
    
    def reset(self, seed=0):
        # Initialize rng_key and state arrays
        x_key = jax.random.PRNGKey(seed)
        agents_state = np.zeros((self.n_agents, self.agents[0].num_vars))
        # Reset the patch
        patch_state = self.patch.reset()
        # Generate random position for each agent
        for i,agent in enumerate(self.agents):
            x_key, y_key = jax.random.split(x_key)
            x = jax.random.uniform(x_key, minval=0,maxval=self.x_max)
            y = jax.random.uniform(y_key, minval=0,maxval=self.y_max)
            agents_state[i,:] = agent.reset(x,y)
            while agent.is_in_patch(agents_state[i], patch_state):
                x_key, y_key = jax.random.split(x_key)
                x = jax.random.uniform(x_key, minval=0,maxval=self.x_max)
                y = jax.random.uniform(y_key, minval=0,maxval=self.y_max)
                agents_state[i,:] = agent.reset(x,y)
        # Reset counter
        step_idx = 0
        # Store agents' observations
        agents_obs = self.get_states(agents_state, patch_state)
        # Return the states and observations
        env_state = (agents_state, patch_state, step_idx) 
        return env_state, agents_obs

    def step(self, env_state, *actions):
        (agents_state, patch_state, step_idx) = env_state
        rewards = list(range(agents_state.shape[0]))
        for i,action in enumerate(actions):
            # Flatten array if needed
            action = action.flatten()
            # Update agent position
            agents_state[i, :] = self.agents[i].update_position(agents_state[i, :], action)
            # Update dynamical system of allocating resources
            agent_state, reward, s_eaten = self.agents[i].update_energy(agents_state[i, :], patch_state, action, dt=0.1/self.n_agents)
            agents_state[i, :] = agent_state
            patch_state = self.patch.update_resources(patch_state, s_eaten, dt=0.1/self.n_agents)
            rewards[i] = reward+self.alpha # Only positive rewards are given this way
        # Apply social welfare function here
        rewards = np.array(rewards)
        #energies = [agent.get_energy() for agent in self.agents]
        social_welfare = self.sw_fun(rewards)
        rewards += social_welfare
        # Update counter
        step_idx += 1
        # Update states AFTER dynamical system updates of each agent have been made
        next_states = self.get_states(agents_state, patch_state)
        # When any of the agents dies, the environment is terminated 
        terminated = np.any(agents_state[:,-1] == 0)
        truncated = step_idx >= self.step_max
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
    
    def update_energy(self,agent_state,patch_state,action,dt=0.1):
        s_eaten = (self.is_in_patch(agent_state,patch_state)).astype(int)*self.beta*patch_state[3]
        action_penalty = np.linalg.norm(action)
        de = dt*(s_eaten - self.alpha*action_penalty)
        # If agent has negative or zero energy, put the energy value at zero and consider the agent dead
        e_agent = agent_state[-1]
        if e_agent + de > 0: 
            agent_state[-1] = e_agent + de
        else:
            de = np.array(0, dtype=jnp.float32) # when dead reward should be/remain zero
        return agent_state, de, s_eaten # reward and amount of resource eaten respectively
        
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
        n_agents = self.env.n_agents
        self.n_agents = n_agents
        self.step_idx = 0
        self.agent_poss = jnp.empty((self.env.step_max, n_agents, 2))
        self.es_agent = jnp.empty((self.env.step_max, n_agents))
        self.rewards = jnp.empty((self.env.step_max, n_agents))
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
        self.es_agent = self.es_agent.at[step_idx-1].set(agents_state[:,-1])
        self.rewards = self.rewards.at[step_idx-1].set([reward.item() for reward in rewards])
        return (env_state, next_ss, rewards, terminated, truncated, info)

    def render(self, save=True, path=""):
        FPS = 15
        # Generate GIF filename
        n_agents = self.env.n_agents
        fname = f"{n_agents}_agents_one_patch.gif"
        fname = os.path.join(path, fname)
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
        [plt.plot(self.rewards[:,i], label=f"Agent {i+1}") for i in range(self.rewards.shape[1])]
        if self.n_agents > 1:
            plt.legend()
        plt.savefig(os.path.join(path, "agent_rewards.png"))
