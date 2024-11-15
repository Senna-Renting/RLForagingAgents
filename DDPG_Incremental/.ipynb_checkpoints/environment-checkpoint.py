import jax
import jax.numpy as jnp
from diffrax import ODETerm, Tsit5

class Environment:
    def __init__(self, seed=0, patch_radius=1, agent_step=0.2, s_init=10, e_init=3, e_max=40, beta=1, eta=11, gamma=1.1, eps=0.1, step_max=200, x_max=5, y_max=5):
        self.e_agent = jnp.array(e_init, dtype=jnp.float32)
        self.agent_pos = jnp.array([x_max/2, y_max/2]) # Just a dummy position as it will be modified by reset function
        self.patch_pos = jnp.array([x_max/2, y_max/2])
        self.patch_radius = patch_radius 
        self.s_patch = jnp.array(s_init, dtype=jnp.float32) # start amount of resources in patch
        self.e_max = e_max # max amount of energy allowed to be in agent
        self.beta = beta # eating rate
        self.eta = eta # growth rate
        self.gamma = gamma # decay rate
        self.eps = eps # agent energy depletion rate
        self.agent_step = agent_step
        self.step_idx = 0
        self.step_max = step_max
        self.x_max = x_max
        self.y_max = y_max
        # ODE solver initialization for resource + energy updates
        self.solver = Tsit5()
        self.solver_state = self.solver.init(ODETerm(self.term), self.step_idx, self.step_idx+1, (self.s_patch, self.e_agent), None)
    
    def reset(self, seed=0):
        # Put the agent randomly somewhere outside the patch in the environment (uniform rejection sampling)
        x_key, y_key = jax.random.split(jax.random.PRNGKey(seed))
        self.agent_pos = jnp.array([jax.random.uniform(x_key, minval=0,maxval=self.x_max), jax.random.uniform(y_key, minval=0,maxval=self.y_max)])
        while not jnp.linalg.norm(self.agent_pos - self.patch_pos) > self.patch_radius:
            x_key, y_key = jax.random.split(x_key)
            self.agent_pos = jnp.array([jax.random.uniform(x_key, minval=0,maxval=self.x_max), jax.random.uniform(y_key, minval=0,maxval=self.y_max)])
        # Reset counter
        self.step_idx = 0
        # Return the state
        state = self._get_state()
        # Initialize ODE solver
        self.solver_state = self.solver.init(ODETerm(self.term), self.step_idx, self.step_idx+1, (self.s_patch, self.e_agent), None)
        return state, None # None is to conform with the output of the gym API

    def get_action_space(self):
        action_dim = 1 # Only direction is used (might extend later by adding velocity as a parameter)
        action_range = [-jnp.pi,jnp.pi]
        return action_dim, action_range

    def _get_state(self):
        state = jnp.concatenate([self.agent_pos, jnp.array([self._dist_to_patch()]), jnp.array([self.s_patch])])
        state = jnp.expand_dims(state, 0) # Ensure correct vector shape
        return state
    
    def get_state_space(self):
        state = self._get_state()
        return state.shape
    
    def _dist_to_patch(self):
        return jnp.linalg.norm(self.agent_pos - self.patch_pos)

    def term(self, time, state, args):
        s_patch, e_agent = state
        dist = self._dist_to_patch()
        # Compute patch resource change
        s_eaten = (dist <= self.patch_radius).astype(int)*jnp.min(jnp.array([self.beta*s_patch, self.e_max - e_agent]))
        s_growth = jnp.multiply(self.eta,s_patch)
        s_decay = jnp.multiply(self.gamma,jnp.power(s_patch,2))
        ds = s_growth - s_decay - s_eaten
        # Compute agent energy change
        e_decay = jnp.multiply(self.eps,jnp.power(e_agent,2))
        de = s_eaten - e_decay
        return (ds, de)
    
    def step(self, action):
        # Update location agent
        x = (self.agent_pos.at[0].get() + self.agent_step*jnp.cos(action)).item()
        x = max(0,min(x, self.x_max)) # Update as long as x is between [0,x_max]
        y = (self.agent_pos.at[1].get() + self.agent_step*jnp.sin(action)).item()
        y = max(0,min(y, self.y_max)) # Update as long as y is between [0,y_max]
        self.agent_pos = jnp.array([x,y])
        (ds, de) = self.term(0,(self.s_patch, self.e_agent), None)
        print("Term results: ", ds, de)
        print("State: ", self.s_patch, self.e_agent)
        # Update the ODE solver
        (self.s_patch, self.e_agent), _, _, self.solver_state, _ = self.solver.step(ODETerm(self.term), self.step_idx, self.step_idx+1, (self.s_patch, self.e_agent), None, self.solver_state, made_jump=False)
        # Update counter
        self.step_idx += 1
        # Return the values needed for RL similar to the OpenAI gym implementation (next_state, reward, terminated, truncated)
        next_state = self._get_state()
        reward = de 
        terminated = self.e_agent == 0
        truncated = self.step_idx >= self.step_max
        return next_state, reward, terminated, truncated, None # None is to have similar output shape as gym API
        

