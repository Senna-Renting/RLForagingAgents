import jax
import jax.numpy as jnp
import pygame
from pygame.locals import *
import numpy as np
import matplotlib.pyplot as plt
from pygame_recorder import PygameRecord
import os
from abc import ABC, abstractmethod
    

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
    def _get_state(self):
        ...
    @abstractmethod
    def get_state_space(self):
        ...
    

class OneAgentEnv(Environment):
    def __init__(self, seed=0, patch_radius=2, s_init=10, e_init=1, eta=0.1, beta=0.5, alpha=0.1, gamma=0.01, step_max=400, x_max=5, y_max=5, v_max=0.1):
        super().__init__(seed=seed, patch_radius=patch_radius, s_init=s_init, e_init=e_init, eta=eta, beta=beta, alpha=alpha, gamma=gamma, step_max=step_max, x_max=x_max, y_max=y_max, v_max=v_max)
        self.agent = Agent(0,0,x_max,y_max,e_init,v_max, alpha=alpha, beta=beta)
    
    def reset(self, seed=0):
        # Put the agent randomly somewhere outside the patch in the environment (uniform rejection sampling)
        x_key, y_key = jax.random.split(jax.random.PRNGKey(seed))
        x = jax.random.uniform(x_key, minval=0,maxval=self.x_max)
        y = jax.random.uniform(y_key, minval=0,maxval=self.y_max)
        while self.agent.is_in_patch(self.patch):
            x_key, y_key = jax.random.split(x_key)
            x = jax.random.uniform(x_key, minval=0,maxval=self.x_max)
            y = jax.random.uniform(y_key, minval=0,maxval=self.y_max)
        self.agent.reset(x,y)
        # Reset counter
        self.step_idx = 0
        # Reset the states
        self.patch.reset()
        # Return the state
        state = self._get_state()
        return state, None # None is to conform with the output of the gym API

    def _get_state(self):
        # Select first agent for now
        state = jnp.concatenate([self.agent.get_position(), self.patch.get_position(), jnp.array([self.patch.get_resources(), self.agent.get_energy()])])
        state = jnp.expand_dims(state, 0) # Ensure correct vector shape
        return state

    def get_state_space(self):
        state = self._get_state()
        return state.shape
    
    def step(self, action):
        # Flatten array if needed
        action = action.flatten()
        # Update agent position
        agent_pos = self.agent.update_position(action)
        # Update dynamical system of allocating resources
        reward, s_eaten = self.agent.update_energy(self.patch)
        patch_resources = self.patch.update_resources(s_eaten)
        # Update counter
        self.step_idx += 1
        # Return the values needed for RL similar to the OpenAI gym implementation (next_state, reward, terminated, truncated)
        next_state = self._get_state()
        # Termination is not really needed atm
        terminated = self.agent.get_energy().item() == 0
        truncated = self.step_idx >= self.step_max
        return next_state, reward, terminated, truncated, None # None is to have similar output shape as gym API

    def render_info(self):
        # Return all info useful for rendering
        agent_e = self.agent.get_energy()
        agent_pos = self.agent.get_position()
        return self.x_max, self.y_max, agent_pos, self.patch.get_position(), self.patch.get_radius(), agent_e, self.patch.get_resources()

# TODO: make a two (or N-) agent version of the single-patch foraging environment
class NAgentsEnv(Environment):
    def __init__(self, seed=0, patch_radius=2, s_init=10, e_init=1, eta=0.1, beta=0.5, alpha=0.1, gamma=0.01, step_max=400, x_max=5, y_max=5, v_max=0.1, n_agents=2, sw_fun=lambda x:0):
        super().__init__(seed=seed, patch_radius=patch_radius, s_init=s_init, e_init=e_init, eta=eta, beta=beta, alpha=alpha, gamma=gamma, step_max=step_max, x_max=x_max, y_max=y_max, v_max=v_max)
        self.agents = [Agent(0,0,x_max,y_max,e_init,v_max, alpha=alpha, beta=beta) for i in range(n_agents)]
        self.n_agents = n_agents
        self.sw_fun = sw_fun
        self.alpha = alpha
        self.e_init = e_init
    
    def get_num_agents(self):
        return self.n_agents
    
    def get_state_space(self):
        state = self._get_state(0)
        return state.shape

    def render_info(self):
        # Return all info useful for rendering
        agent_energies = [agent.get_energy() for agent in self.agents]
        agent_positions = [agent.get_position() for agent in self.agents]
        return self.x_max, self.y_max, agent_positions, self.patch.get_position(), self.patch.get_radius(), agent_energies, self.patch.get_resources()
    
    # Select state of agent by index
    def _get_state(self, i):
        # Select first agent for now
        state = jnp.concatenate([self.agents[i].get_position(), self.patch.get_position(), jnp.array([self.patch.get_resources(), self.agents[i].get_energy()])])
        state = jnp.expand_dims(state, 0) # Ensure correct vector shape
        return state
    
    def reset(self, seed=0):
        # Put the agent randomly somewhere outside the patch in the environment (uniform rejection sampling)
        x_key, y_key = jax.random.split(jax.random.PRNGKey(seed))
        for agent in self.agents:
            x_key, y_key = jax.random.split(x_key)
            x = jax.random.uniform(x_key, minval=0,maxval=self.x_max)
            y = jax.random.uniform(y_key, minval=0,maxval=self.y_max)
            agent.reset(x,y)
            while agent.is_in_patch(self.patch):
                x_key, y_key = jax.random.split(x_key)
                x = jax.random.uniform(x_key, minval=0,maxval=self.x_max)
                y = jax.random.uniform(y_key, minval=0,maxval=self.y_max)
                agent.reset(x,y)
        # Reset counter
        self.step_idx = 0
        # Reset the states
        self.patch.reset()
        # Return the state
        states = [self._get_state(i) for i in range(self.n_agents)] 
        return states, None # None is to conform with the output of the gym API

    def step(self, *actions):
        next_states = []
        rewards = []
        for i,action in enumerate(actions):
            # Flatten array if needed
            action = action.flatten()
            # Update agent position
            agent_pos = self.agents[i].update_position(action)
            # Update dynamical system of allocating resources
            reward, s_eaten = self.agents[i].update_energy(self.patch)
            self.patch.update_resources(s_eaten, dt=0.1/self.n_agents)
            rewards.append(reward+self.alpha) # Only positive rewards are given this way
        # Apply social welfare function here
        rewards = np.array(rewards)
        #energies = [agent.get_energy() for agent in self.agents]
        social_welfare = self.sw_fun(rewards)
        rewards += social_welfare
        # Update counter
        self.step_idx += 1
        # Update states AFTER dynamical system updates of each agent have been made
        next_states = [self._get_state(i) for i in range(self.n_agents)]
        # When any of the agents dies, the environment is terminated 
        terminated = np.any([self.agents[i].get_energy().item() == 0 for i in range(self.n_agents)])
        truncated = self.step_idx >= self.step_max
        return next_states, (rewards, social_welfare), terminated, truncated, None # None is to have similar output shape as gym API



class Agent:
    def __init__(self,x,y,x_max,y_max,e_init,v_max, alpha=0.4, beta=0.25):
        # Hyperparameters of dynamical system
        self.alpha = alpha # penalizing coëfficient for movement
        self.beta = beta # amount of eating per timestep
        # General variables
        self.agent_pos = jnp.array([x,y,0,0]) # Structure: [x,y,x_dot,y_dot]
        self.agent_x_dot = 0 # velocity on x-axis
        self.agent_y_dot = 0 # velocity on y-axis
        self.action = jnp.array([0,0])
        self.v_max = v_max
        self.x_max = x_max
        self.y_max = y_max
        self.e_init = e_init
        self.e_agent = jnp.array(self.e_init, dtype=jnp.float32) # resources/energy inside the agent 

    def reset(self,x,y,seed=0):
        v_key, theta_key = jax.random.split(jax.random.PRNGKey(seed))
        self.agent_pos = jnp.array([x,y,0,0])
        self.agent_x_dot = jax.random.uniform(v_key, minval=-self.v_max, maxval=self.v_max)
        self.agent_y_dot = jax.random.uniform(v_key, minval=-self.v_max, maxval=self.v_max)
        self.e_agent = jnp.array(self.e_init, dtype=jnp.float32)
    
    def get_energy(self):
        return self.e_agent
    
    def is_in_patch(self,patch):
        patch_pos = patch.get_position()
        x_diff = self.agent_pos.at[0].get() - patch_pos.at[0].get()
        y_diff = self.agent_pos.at[1].get() - patch_pos.at[1].get()
        dist = jnp.sqrt(x_diff**2 + y_diff**2)
        return dist <= patch.get_radius()
    
    def update_energy(self,patch, dt=0.1):
        s_eaten = (self.is_in_patch(patch)).astype(int)*self.beta*patch.get_resources().item()
        action_penalty = jnp.linalg.norm(self.action)
        de = dt*(s_eaten - self.alpha*action_penalty)
        # If agent has negative or zero energy, put the energy value at zero and consider the agent dead
        if self.e_agent.item() + de > 0: 
            self.e_agent = jnp.array(self.e_agent.item() + de, dtype=jnp.float32)
        else:
            de = jnp.array(0, dtype=jnp.float32) # when dead reward should be/remain zero
            self.e_agent = jnp.array(0, dtype=jnp.float32)
        return de, s_eaten # reward and amount of resource eaten respectively

    def get_position(self):
        return self.agent_pos
        
    def update_position(self,action):
        # Update current action stored by agent
        self.action = action
        # Functions needed to bound the allowed actions
        v_bounded = lambda v: max(min(v, self.v_max), -self.v_max)
        # Compute action values
        self.agent_x_dot = v_bounded(self.agent_x_dot + action.at[0].get())
        self.agent_y_dot = v_bounded(self.agent_y_dot + action.at[1].get())
        # Update position (only if not dead)
        if self.e_agent.item() > 0:
            x = (self.agent_pos.at[0].get() + self.agent_x_dot) % self.x_max
            y = (self.agent_pos.at[1].get() + self.agent_y_dot) % self.y_max
            self.agent_pos = jnp.array([x,y,self.agent_x_dot,self.agent_y_dot])
        #print(self.agent_pos)
        return self.agent_pos

class Patch:
    def __init__(self, x,y,radius,s_init, eta=0.1, gamma=0.01):
        # Hyperparameters of dynamical system
        self.eta = eta # regeneration rate of resources
        self.gamma = gamma # decay rate of resources
        # General variables
        self.pos = jnp.array([x,y])
        self.radius = radius
        self.s_init = s_init
        self.resources = jnp.array(self.s_init, dtype=jnp.float32)
    def get_resources(self):
        return self.resources
    def get_position(self):
        return self.pos
    def get_radius(self):
        return self.radius
    def update_resources(self, eaten, dt=0.1):
        s_growth = jnp.multiply(self.eta,self.resources)
        s_decay = jnp.multiply(self.gamma,jnp.power(self.resources,2))
        ds = dt*(s_growth - s_decay - eaten)
        self.resources = jnp.array(max(0,self.resources.item() + ds), dtype=jnp.float32)
        return self.resources
    def reset(self):
        self.resources = jnp.array(self.s_init, dtype=jnp.float32)

# TODO: make general for N-agent environments
class RenderOneAgentEnvironment:
    def __init__(self, env):
        self.env = env
        self.closed = False
        # Tracking arrays
        self.agent_poss = np.empty((self.env.step_max, 2))
        self.ss_patch = np.empty(self.env.step_max)
        self.es_agent = np.empty(self.env.step_max)
        self.rewards = np.empty(self.env.step_max)
    
    def reset(self, seed=0):
        results = self.env.reset(seed=seed)
        size_x, size_y, agent_pos, patch_pos, patch_radius, e_agent, s_patch = self.env.render_info()
        self.agent_poss[self.env.step_idx-1] = [agent_pos.at[0].get(), agent_pos.at[1].get()]
        self.ss_patch[self.env.step_idx-1] = s_patch.item()
        return results
    
    def step(self, action):
        (next_s, reward, terminated, truncated, info) = self.env.step(action)
        size_x, size_y, agent_pos, patch_pos, patch_radius, e_agent, s_patch = self.env.render_info()
        self.agent_poss[self.env.step_idx-1] = [agent_pos.at[0].get(), agent_pos.at[1].get()]
        self.ss_patch[self.env.step_idx-1] = s_patch.item()
        self.es_agent[self.env.step_idx-1] = e_agent.item()
        self.rewards[self.env.step_idx-1] = reward.item()
        return (next_s, reward, terminated, truncated, info)

    def render(self, save=True, path=""):
        FPS = 20
        # Generate unique GIF filename
        fname = "one_agent_one_patch.gif"
        i = 0
        while os.path.isfile(fname):
            i += 1
            fname = f"one_agent_one_patch-{i}.gif"
        # Render game and simultaneously save it as a gif
        with PygameRecord(fname, FPS) as recorder:
            # Pygame screen init
            pygame.init()
            clock = pygame.time.Clock()
            # Get render info
            size_x, size_y, *_ = self.env.render_info()
            scale = 100
            agent_size = size_x/scale
            screen_x = scale*size_x
            screen_y = scale*size_y
            screen = pygame.display.set_mode((screen_x, screen_y))
            red_color = pygame.Color(255, 0, 0)
            white_color = pygame.Color(255,255,255)
            # Select used segments of tracking arrays
            self.agent_poss = self.agent_poss[:self.env.step_idx - 1]
            self.ss_patch = self.ss_patch[:self.env.step_idx - 1]
            self.es_agent = self.es_agent[:self.env.step_idx - 1]
            self.rewards = self.rewards[:self.env.step_idx - 1]
            # Render on initialization
            for i in range(self.env.step_idx - 2):
                for event in pygame.event.get():
                    if event.type == QUIT:
                        return
                clock.tick(FPS)
                # Get render info
                size_x, size_y, agent_pos, patch_pos, patch_radius, e_agent, s_patch = self.env.render_info()
                #print(self.agent_poss[i])
                max_ratio_patch = self.ss_patch[i]/(self.env.eta/self.env.gamma)
                patch_color = pygame.Color(0, max(50,int(max_ratio_patch*255)), 0)
                agent_pos = [scale*self.agent_poss[i, 0], scale*self.agent_poss[i, 1]]
                patch_pos = [scale*patch_pos.at[0].get().item(), scale*patch_pos.at[1].get().item()]
                # Draw information on screen
                screen.fill(white_color)
                pygame.draw.circle(screen, patch_color, patch_pos, scale*patch_radius)
                pygame.draw.circle(screen, red_color, agent_pos, scale*agent_size)
                # Update screen
                pygame.display.update()
                recorder.add_frame()
            # Close the screen
            pygame.quit()
            if save:
                recorder.save()
        # Show the plots for the agent and the patch resources
        fig, (ax1, ax2, ax3) = plt.subplots(3)
        ax1.set_title("Energy of agent")
        ax1.set_xlabel("Timestep")
        ax1.set_ylabel("Energy")
        ax1.plot(self.es_agent)
        ax2.set_title("Resources in patch")
        ax2.set_xlabel("Timestep")
        ax2.set_ylabel("Resources")
        ax2.plot(self.ss_patch)
        ax3.set_title("Rewards collected at timesteps")
        ax3.set_xlabel("Timestep")
        ax3.set_ylabel("Reward value")
        ax3.plot(self.rewards)
        plt.tight_layout()
        plt.show()

class RenderNAgentsEnvironment:
    def __init__(self, env):
        self.env = env
        self.closed = False
        # Tracking arrays
        n_agents = self.env.get_num_agents()
        self.n_agents = n_agents
        self.agent_poss = np.empty((self.env.step_max, n_agents, 2))
        self.es_agent = np.empty((self.env.step_max, n_agents))
        self.rewards = np.empty((self.env.step_max, n_agents))
        self.ss_patch = np.empty(self.env.step_max)
    
    def reset(self, seed=0):
        results = self.env.reset(seed=seed)
        size_x, size_y, agents_pos, patch_pos, patch_radius, e_agents, s_patch = self.env.render_info()
        self.agent_poss[self.env.step_idx-1] = np.array([[agent_pos.at[0].get(), agent_pos.at[1].get()] for agent_pos in agents_pos])
        self.ss_patch[self.env.step_idx-1] = s_patch.item()
        return results
    
    def step(self, *actions):
        (next_ss, (rewards, social_welfare), terminated, truncated, info) = self.env.step(*actions)
        size_x, size_y, agents_pos, patch_pos, patch_radius, e_agents, s_patch = self.env.render_info()
        self.agent_poss[self.env.step_idx-1] = np.array([[agent_pos.at[0].get(), agent_pos.at[1].get()] for agent_pos in agents_pos])
        self.ss_patch[self.env.step_idx-1] = s_patch.item()
        self.es_agent[self.env.step_idx-1] = np.array([e_agent.item() for e_agent in e_agents])
        self.rewards[self.env.step_idx-1] = np.array([reward.item() for reward in rewards])
        return (next_ss, rewards, terminated, truncated, info)

    def render(self, save=True, path=""):
        FPS = 20
        # Generate GIF filename
        n_agents = self.env.get_num_agents()
        fname = f"{n_agents}_agents_one_patch.gif"
        fname = os.path.join(path, fname)
        # Render game and simultaneously save it as a gif
        with PygameRecord(fname, FPS) as recorder:
            # Pygame screen init
            pygame.init()
            clock = pygame.time.Clock()
            # Get render info
            size_x, size_y, *_ = self.env.render_info()
            scale = 100
            agent_size = size_x/scale
            screen_x = scale*size_x
            screen_y = scale*size_y
            screen = pygame.display.set_mode((screen_x, screen_y))
            red_color = pygame.Color(255, 0, 0)
            white_color = pygame.Color(255,255,255)
            # Select used segments of tracking arrays
            self.agent_poss = self.agent_poss[:self.env.step_idx - 1]
            self.ss_patch = self.ss_patch[:self.env.step_idx - 1]
            self.es_agent = self.es_agent[:self.env.step_idx - 1]
            self.rewards = self.rewards[:self.env.step_idx - 1]
            # Render on initialization
            for i in range(self.env.step_idx - 2):
                for event in pygame.event.get():
                    if event.type == QUIT:
                        return
                clock.tick(FPS)
                # Get render info
                size_x, size_y, agents_pos, patch_pos, patch_radius, e_agents, s_patch = self.env.render_info()
                max_ratio_patch = self.ss_patch[i]/(self.env.eta/self.env.gamma)
                #print("Patch color ratio: ",max_ratio_patch)
                patch_color = pygame.Color(0, max(50,int(max_ratio_patch*255)), 0)
                agents_pos = scale*self.agent_poss[i]
                patch_pos = [scale*patch_pos.at[0].get().item(), scale*patch_pos.at[1].get().item()]
                # Draw information on screen
                screen.fill(white_color)
                pygame.draw.circle(screen, patch_color, patch_pos, scale*patch_radius)
                [pygame.draw.circle(screen, red_color, agent_pos, scale*agent_size) for agent_pos in agents_pos]
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
        
        

