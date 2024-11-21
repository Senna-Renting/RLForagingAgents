import jax
import jax.numpy as jnp
import pygame
from pygame.locals import *
import numpy as np

class Environment:
    def __init__(self, seed=0, patch_radius=1, s_init=10, e_init=1, eta=0.01, beta=0.5, alpha=0.5, gamma=0.001, step_max=200, x_max=5, y_max=5, v_max=1):
        self.e_agent = jnp.array(e_init, dtype=jnp.float32)
        self.agent_pos = jnp.array([x_max/2, y_max/2, 0, 0]) # Just a dummy position as it will be modified by reset function
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
        self.agent_pos = jnp.array([jax.random.uniform(x_key, minval=0,maxval=self.x_max), jax.random.uniform(y_key, minval=0,maxval=self.y_max), 0,0])
        while not self._dist_to_patch() > self.patch_radius:
            x_key, y_key = jax.random.split(x_key)
            self.agent_pos = jnp.array([jax.random.uniform(x_key, minval=0,maxval=self.x_max), jax.random.uniform(y_key, minval=0,maxval=self.y_max), 0,0])
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
        action_range = [-2,2]
        return action_dim, action_range

    def _get_state(self):
        state = jnp.concatenate([self.agent_pos, jnp.array([self._dist_to_patch()]), jnp.array([self.s_patch])])
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

    def term(self, h=0.01):
        dist = self._dist_to_patch()
        # Compute patch resource change
        s_eaten = (dist <= self.patch_radius).astype(int)*self.beta*self.s_patch
        s_growth = jnp.multiply(self.eta,self.s_patch)
        s_decay = jnp.multiply(self.gamma,jnp.power(self.s_patch,2))
        ds = h*(s_growth - s_decay - s_eaten)
        # Compute agent energy change
        v_max_norm = jnp.sqrt(self.v_max**2 + self.v_max**2)
        v_norm = jnp.linalg.norm(jnp.array([self.agent_pos.at[2].get(), self.agent_pos.at[3].get()]))
        v_norm /= v_max_norm
        e_decay = self.alpha*self.e_agent
        de = h*(s_eaten - e_decay*v_norm) 
        return (ds, de)

    
    # TODO: Implement the action as a 2D acceleration term
    def step(self, action, h=0.1):
        # Update velocity of agent (bounded by v_max)
        v_bounded = lambda v: min(self.v_max + v, 2*self.v_max) - self.v_max
        #print("Action: ", action)
        x_dot = v_bounded((self.agent_pos.at[2].get() + h**2*action.at[0,0].get()).item())
        y_dot = v_bounded((self.agent_pos.at[3].get() + h**2*action.at[0,1].get()).item())
        #print("Step info: ", x_dot, " : ", action)
        # Update location agent
        x = (self.agent_pos.at[0].get() + h*x_dot).item() % self.x_max
        y = (self.agent_pos.at[1].get() + h*y_dot).item() % self.y_max
        self.agent_pos = jnp.array([x,y,x_dot,y_dot])
        (ds, de) = self.term()
        # Update patch and agent state
        self.s_patch = self.s_patch + ds
        self.e_agent = self.e_agent + de
        # Update counter
        self.step_idx += 1
        # Return the values needed for RL similar to the OpenAI gym implementation (next_state, reward, terminated, truncated)
        next_state = self._get_state()
        reward = de 
        terminated = self.e_agent == 0
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
        self.agent_poss[self.env.step_idx-1] = [agent_pos.at[0].get(), agent_pos.at[1].get()]
        self.ss_patch[self.env.step_idx-1] = s_patch.item()
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
        # Render on initialization
        for i in range(self.env.step_max):
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
        
        

