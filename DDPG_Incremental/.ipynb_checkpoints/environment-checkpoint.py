# TODO: Code the rectangular patch environments, in which agents can roam around
class Environment:
    def __init__(self, seed=0, patch_radius=1, s_init=10, e_max=10, beta=11, eta=11, gamma=1.1, eps=0.01, step_max=200, x_max=5, y_max=5):
        self.e_agent = e_init
        self.agent_pos = jnp.array([x_max/2, y_max/2]) # Just a dummy position as it will be modified by reset function
        self.patch_pos = jnp.array([x_max/2, y_max/2])
        self.patch_radius = patch_radius 
        self.s_patch = s_init # start amount of resources in patch
        self.e_max = e_max # max amount of energy allowed to be in agent
        self.beta = beta # eating rate
        self.eta = eta # growth rate
        self.gamma = gamma # decay rate
        self.eps = eps # agent energy depletion rate
        self.step_idx = 0
        self.step_max = step_max
        self.x_max = x_max
        self.y_max = y_max
    def reset(self, seed=0):
        # Put the agent randomly somewhere outside the patch in the environment (uniform rejection sampling)
        x_key, y_key = jax.random.split(jax.random.PRNGKey(seed))
        self.agent_pos = jnp.array([jax.random.uniform(x_key, minval=0,maxval=self.x_max), jax.random.uniform(y_key, minval=0,maxval=self.y_max)])
        while not jnp.linalg.norm(self.agent_pos - self.patch_pos) > self.patch_radius:
            x_key, y_key = jax.random.split(x_key)
            self.agent_pos = jnp.array([jax.random.uniform(x_key, minval=0,maxval=self.x_max), jax.random.uniform(y_key, minval=0,maxval=self.y_max)])
        
        # Return the state
        state = jnp.concatenate(self.agent_pos, dist, self.s_patch)
        return state, None # None is to conform with the output of the gym API

    def step(self, action):
        # Update location agent
        x = jnp.max([0,jnp.min([self.agent_pos.at[0] + jnp.cos(action), self.x_max])) # Update as long as x is between [0,x_max]
        y = jnp.max([0,jnp.min([self.agent_pos.at[1] + jnp.sin(action), self.y_max])) # Update as long as y is between [0,y_max]
        self.agent_pos = jnp.array([x,y])
        # Compute distance to patch
        dist = jnp.linalg.norm(self.agent_pos - self.patch_pos)
        # Update resources in patch
        s_eaten = 0
        if dist <= self.patch_radius:
            s_eaten = jnp.min([self.beta*self.s_patch, self.e_max - self.e_agent])
        s_growth = self.eta*self.s_patch
        s_decay = self.gamma*self.s_patch**2
        ds = s_growth - s_decay - s_eaten
        self.s_patch += ds
        # Update resources in agent
        e_decay = self.eps*self.e_agent**2
        de = s_eaten - e_decay
        self.e_agent += de
        # Update counter
        self.step_idx += 1
        # Return the values needed for RL similar to the OpenAI gym implementation (next_state, reward, terminated, truncated)
        next_state = jnp.concatenate(self.agent_pos, dist, self.s_patch)
        reward = self.e_agent
        terminated = reward == 0
        truncated = self.step_idx == self.step_max
        return next_state, reward, terminated, truncated
        

