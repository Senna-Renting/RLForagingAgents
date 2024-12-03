import jax
import jax.numpy as jnp
import pygame
from pygame.locals import *
import numpy as np
import matplotlib.pyplot as plt

class Environment:
    def __init__(self, seed=0, patch_radius=2, s_init=4, e_init=1, eta=0.1, beta=0.25, alpha=0.4, gamma=0.01, step_max=400, x_max=5, y_max=5, v_max=0.1):
        self.e_agent = jnp.array(e_init, dtype=jnp.float32)
        self.agent_pos = jnp.array([x_max/2, y_max/2, 0, 0]) # Just a dummy position as it will be modified by reset function
        # TODO: Rewrite agent's actions in environment as polar coördinate updates
        self.agent_v = 0
        self.agent_v_dot = 0
        self.agent_theta = 0
        self.agent_theta_dot = 0
        self.patch_pos = jnp.array([x_max/2, y_max/2])
        self.patch_radius = patch_radius 
        self.s_init = s_init
        self.e_init = e_init
        self.s_patch = jnp.array(s_init, dtype=jnp.float32) # start amount of resources in patch
        self.beta = beta # energy uptake regularizer based on resource amount for agent
        self.eta = eta # growth rate
        self.gamma = gamma # decay rate
        self.alpha = alpha # energy decay of agent
        self.step_idx = 0
        self.step_max = step_max
        self.x_max = x_max
        self.y_max = y_max
        self.v_max = v_max
    
    def reset(self, seed=0):
        # Put the agent randomly somewhere outside the patch in the environment (uniform rejection sampling)
        x_key, y_key = jax.random.split(jax.random.PRNGKey(seed))
        self.agent_pos = jnp.array([jax.random.uniform(x_key, minval=0,maxval=self.x_max), jax.random.uniform(y_key, minval=0,maxval=self.y_max),0,0])
        while not self._dist_to_patch() > self.patch_radius:
            x_key, y_key = jax.random.split(x_key)
            self.agent_pos = jnp.array([jax.random.uniform(x_key, minval=0,maxval=self.x_max), jax.random.uniform(y_key, minval=0,maxval=self.y_max),0,0])
        v_key, theta_key = jax.random.split(x_key)
        self.agent_v = jax.random.uniform(v_key, minval=-self.v_max, maxval=self.v_max)
        self.agent_v_dot = 0
        self.agent_theta = jax.random.uniform(theta_key, minval=-jnp.pi, maxval=jnp.pi)
        self.agent_theta_dot = 0
        # Reset counter
        self.step_idx = 0
        # Reset the states
        self.e_agent = jnp.array(self.e_init, dtype=jnp.float32)
        self.s_patch = jnp.array(self.s_init, dtype=jnp.float32)
        # Return the state
        state = self._get_state()
        return state, None # None is to conform with the output of the gym API

    def get_action_space(self):
        action_dim = 2 # We use two acceleration terms (one for x and one for y)
        action_range = [jnp.pi/10, self.v_max]
        return action_dim, action_range

    def _get_state(self):
        state = jnp.concatenate([self.agent_pos, jnp.array([self.s_patch, self.e_agent])])
        #state = self.agent_pos
        state = jnp.expand_dims(state, 0) # Ensure correct vector shape
        return state
    
    def get_state_space(self):
        state = self._get_state()
        #print(state.shape)
        return state.shape

    def render_info(self):
        # Return all info useful for rendering
        return self.x_max, self.y_max, self.agent_pos, self.patch_pos, self.patch_radius, self.e_agent, self.s_patch
    
    def _dist_to_patch(self):
        x_diff = self.agent_pos.at[0].get() - self.patch_pos.at[0].get()
        y_diff = self.agent_pos.at[1].get() - self.patch_pos.at[1].get()
        return jnp.sqrt(x_diff**2 + y_diff**2)

    def term(self, h=0.1):
        dist = self._dist_to_patch()
        # Compute patch resource change
        s_eaten = (dist <= self.patch_radius).astype(int)*self.beta*self.s_patch.item()
        s_growth = jnp.multiply(self.eta,self.s_patch)
        s_decay = jnp.multiply(self.gamma,jnp.power(self.s_patch,2))
        ds = h*(s_growth - s_decay - s_eaten)
        # Compute agent energy change
        v_max_norm = jnp.sqrt(self.v_max**2 + self.v_max**2)
        v_norm = jnp.linalg.norm(jnp.array([self.agent_pos.at[2].get(), self.agent_pos.at[3].get()]))
        v_norm /= v_max_norm
        de = h*(s_eaten - self.alpha*v_norm) 
        return (ds, de)

    
    # TODO: Implement the action as a 2D acceleration term
    def step(self, action, h=0.1):
        # Flatten array if needed
        action = action.flatten()
        # Update velocity of agent (bounded by v_max)
        theta_max = jnp.pi/10
        theta_bounded = lambda theta: max(min(theta, theta_max), -theta_max)
        v_bounded = lambda v: max(min(v, self.v_max), -self.v_max)
        # Update action params
        damping = 0.95
        self.agent_v = v_bounded(damping*self.agent_v + h*action.at[1].get())
        self.agent_theta = theta_bounded(self.agent_theta + h*action.at[0].get())
        #print(self.agent_v, self.agent_theta)
        # Update location agent
        x_dot = self.agent_v*np.cos(self.agent_theta)
        y_dot = self.agent_v*np.sin(self.agent_theta)
        #print(self.agent_pos, self.agent_pos.shape)
        x = (self.agent_pos.at[0].get() + x_dot) % self.x_max
        y = (self.agent_pos.at[1].get() + y_dot) % self.y_max
        self.agent_pos = jnp.array([x,y,x_dot,y_dot])
        #print(self.agent_pos)
        (ds, de) = self.term()
        # Update patch and agent state
        self.s_patch = self.s_patch + ds
        self.e_agent = jnp.array(max(0.001,(self.e_agent + de).item()))
        # Update counter
        self.step_idx += 1
        # Return the values needed for RL similar to the OpenAI gym implementation (next_state, reward, terminated, truncated)
        next_state = self._get_state()
        reward = de 
        terminated = self.e_agent.item() == 0
        truncated = self.step_idx >= self.step_max
        return next_state, reward, terminated, truncated, None # None is to have similar output shape as gym API

# TODO: Create a render class that decorates the environment, by generating a visual display at each step 
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
        results = self.env.step(action)
        size_x, size_y, agent_pos, patch_pos, patch_radius, e_agent, s_patch = self.env.render_info()
        _, reward = self.env.term()
        self.agent_poss[self.env.step_idx-1] = [agent_pos.at[0].get(), agent_pos.at[1].get()]
        self.ss_patch[self.env.step_idx-1] = s_patch.item()
        self.es_agent[self.env.step_idx-1] = e_agent.item()
        self.rewards[self.env.step_idx-1] = reward.item()
        return results

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
        
        

