import jax
import jax.numpy as jnp
import pygame
from pygame.locals import *
import numpy as np
import matplotlib.pyplot as plt

class Environment:
    def __init__(self, seed=0, patch_radius=2, s_init=4, e_init=1, eta=0.1, beta=0.25, alpha=0.01, gamma=0.01, step_max=400, x_max=5, y_max=5, v_max=0.1):
        self.patch = Patch(x_max/2, y_max/2, patch_radius, s_init, eta=eta, gamma=gamma)
        self.agent = Agent(0,0,x_max,y_max,e_init,v_max, alpha=alpha, beta=beta)
        self.x_max = x_max
        self.y_max = y_max
        self.v_max = v_max
        # Eta and gamma are used for computing the rendering (don't remove them!!)
        self.eta = eta
        self.gamma = gamma
        self.step_max = step_max
        self.step_idx = 0
    
    def reset(self, seed=0):
        # Put the agent randomly somewhere outside the patch in the environment (uniform rejection sampling)
        x_key, y_key = jax.random.split(jax.random.PRNGKey(seed))
        x = jax.random.uniform(x_key, minval=0,maxval=self.x_max)
        y = jax.random.uniform(y_key, minval=0,maxval=self.y_max)
        self.agent.reset(x,y)
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

    def get_action_space(self):
        action_dim = 2 # We use two acceleration terms (one for x and one for y)
        action_range = [jnp.pi/10, self.v_max]
        return action_dim, action_range

    def _get_state(self):
        state = jnp.concatenate([self.agent.get_position(), jnp.array([self.patch.get_resources(), self.agent.get_energy()])])
        #state = self.agent_pos
        state = jnp.expand_dims(state, 0) # Ensure correct vector shape
        return state
    
    def get_state_space(self):
        state = self._get_state()
        #print(state.shape)
        return state.shape

    def render_info(self):
        # Return all info useful for rendering
        return self.x_max, self.y_max, self.agent.get_position(), self.patch.get_position(), self.patch.get_radius(), self.agent.get_energy(), self.patch.get_resources()
    
    # TODO: Implement the action as a 2D acceleration term
    def step(self, action, h=0.1):
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
        terminated = self.agent.get_energy().item() == 0
        truncated = self.step_idx >= self.step_max
        return next_state, reward, terminated, truncated, None # None is to have similar output shape as gym API

# TODO: create an agent class that interacts with the environment and stores information specific to an agent
class Agent:
    def __init__(self,x,y,x_max,y_max,e_init,v_max, alpha=0.1, beta=0.25):
        # Hyperparameters of dynamical system
        self.alpha = alpha # penalizing coëfficient for movement
        self.beta = beta # amount of eating per timestep
        # General variables
        self.agent_pos = jnp.array([x,y,0,0]) # Structure: [x,y,x_dot,y_dot]
        self.agent_theta = 0 # angle of agent
        self.agent_v = 0 # speed of movement (negative means backward movement)
        self.v_max = v_max
        self.x_max = x_max
        self.y_max = y_max
        self.e_init = e_init
        self.e_agent = jnp.array(self.e_init, dtype=jnp.float32) # resources/energy inside the agent 

    def reset(self,x,y,seed=0):
        v_key, theta_key = jax.random.split(jax.random.PRNGKey(seed))
        self.agent_pos = jnp.array([x,y,0,0])
        self.agent_v = jax.random.uniform(v_key, minval=-self.v_max, maxval=self.v_max)
        self.agent_theta = jax.random.uniform(theta_key, minval=-jnp.pi, maxval=jnp.pi)
        self.e_agent = jnp.array(self.e_init, dtype=jnp.float32)
    
    def get_energy(self):
        return self.e_agent
    
    def is_in_patch(self,patch):
        patch_pos = patch.get_position()
        x_diff = self.agent_pos.at[0].get() - patch_pos.at[0].get()
        y_diff = self.agent_pos.at[1].get() - patch_pos.at[1].get()
        dist = jnp.sqrt(x_diff**2 + y_diff**2)
        return dist <= patch.get_radius()
    
    def update_energy(self,patch, dt=0.02):
        s_eaten = (self.is_in_patch(patch)).astype(int)*self.beta*patch.get_resources().item()
        v_norm = jnp.abs(self.agent_v) / self.v_max
        de = dt*(s_eaten - self.alpha*v_norm) 
        self.e_agent = jnp.array(self.e_agent.item() + de, dtype=jnp.float32)
        return de, s_eaten # reward and amount of resource eaten respectively

    def get_position(self):
        return self.agent_pos
        
    def update_position(self,action, dt=0.1):
        # Functions needed to bound the allowed actions
        theta_max = jnp.pi/10
        theta_bounded = lambda theta: max(min(theta, theta_max), -theta_max)
        v_bounded = lambda v: max(min(v, self.v_max), -self.v_max)
        # Compute action values
        damping = 0.95 # Adds friction to the movement of the agent (slows down over time)
        self.agent_v = v_bounded(damping*self.agent_v + dt*action.at[1].get())
        self.agent_theta = theta_bounded(self.agent_theta + dt*action.at[0].get())
        # Update position
        x_dot = self.agent_v*np.cos(self.agent_theta)
        y_dot = self.agent_v*np.sin(self.agent_theta)
        x = (self.agent_pos.at[0].get() + x_dot) % self.x_max
        y = (self.agent_pos.at[1].get() + y_dot) % self.y_max
        self.agent_pos = jnp.array([x,y,x_dot,y_dot])
        return self.agent_pos

# TODO: put relevant code for patch here
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
    def update_resources(self, eaten, dt=0.02):
        s_growth = jnp.multiply(self.eta,self.resources)
        s_decay = jnp.multiply(self.gamma,jnp.power(self.resources,2))
        ds = dt*(s_growth - s_decay - eaten)
        self.resources = jnp.array(self.resources.item() + ds, dtype=jnp.float32)
        return self.resources
    def reset(self):
        self.resources = jnp.array(self.s_init, dtype=jnp.float32)

class RenderEnvironment:
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
        #print(f"Action: {action}")
        (next_s, reward, terminated, truncated, info) = self.env.step(action)
        size_x, size_y, agent_pos, patch_pos, patch_radius, e_agent, s_patch = self.env.render_info()
        self.agent_poss[self.env.step_idx-1] = [agent_pos.at[0].get(), agent_pos.at[1].get()]
        self.ss_patch[self.env.step_idx-1] = s_patch.item()
        self.es_agent[self.env.step_idx-1] = e_agent.item()
        self.rewards[self.env.step_idx-1] = reward.item()
        return (next_s, reward, terminated, truncated, info)

    def render(self):
        # Pygame screen init
        pygame.init()
        clock = pygame.time.Clock()
        # Get render info
        size_x, size_y, *_ = self.env.render_info()
        scale = 100
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
            clock.tick(20)
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
            pygame.draw.circle(screen, red_color, agent_pos, scale*(patch_radius/25))
            # Update screen
            pygame.display.update()
        # Close the screen
        pygame.quit()
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
        
        

