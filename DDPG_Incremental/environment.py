import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os

# Helper function for computing the Nash Social Welfare function (aka geometric mean)
def compute_NSW(rewards):
    NSW = np.power(np.prod(rewards,axis=0), 1/rewards.shape[0])
    return NSW
    
class NAgentsEnv():
    def __init__(self, patch_radius=10,s_init=10, e_init=5, eta=0.1, beta=0.1, env_gamma=0.01, step_max=600, x_max=50, y_max=50, v_max=4, n_agents=2, in_patch_only=False, p_welfare=0, patch_resize=False, seed=0, comm_type=0, msg_type=[], **kwargs):
        # Main variables used in environment
        self.x_max = x_max
        self.y_max = y_max
        self.v_max = v_max
        self.e_max = s_init + e_init
        self.eta = eta
        self.env_gamma = env_gamma
        self.step_max = step_max
        self.step_idx = 0
        self.damping = 0.3
        self.p_still = 0.02
        self.p_act = 0.2
        self.p_att = 0.02
        self.p_comm = 0.04
        self.msg_noise = 0.01
        self.n_agents = n_agents
        self.beta = beta
        self.e_init = e_init
        self.s_init = s_init
        self.in_patch_only = in_patch_only
        self.patch_resize = patch_resize
        self.p_welfare = p_welfare
        self.seed = seed
        self.msg_type = msg_type
        self.comm_type = comm_type
        # Initialize patch and agents
        self.patch = Patch(x_max/2, y_max/2, patch_radius, self)
        self.agents = [Agent(0,0,self,id=i) for i in range(n_agents)]
        # Initialize latest observation states
        self.latest_obs = np.zeros((self.n_agents, self.n_agents, self.agents[0].num_vars-1))
        self.latest_patch = 0

    def get_action_space(self):
        message_dim = len(self.msg_type) > 0 # Add a channel for each message type
        attention_dim = (message_dim > 0) # Add an attention dimension
        action_dim = 2 # We use two acceleration terms (one for x and one for y)
        if self.n_agents == 2:
            action_range = [[-self.v_max,-self.v_max,0,0], [self.v_max,self.v_max,1,1]]
        else:
            action_range = [[-self.v_max, -self.v_max], [self.v_max, self.v_max]]
        return action_dim+message_dim+attention_dim, action_range

    def get_comm_types(self):
        return ["None", "Noisy message", "Full observation", "Noise only"]
    
    def get_msg_types(self):
        return [(1,self.msg_noise*self.e_max,"Energy"), (2,self.msg_noise*self.x_max,"Position"), 
                (2,self.msg_noise*self.v_max,"Velocity"), (2,self.msg_noise*self.v_max,"Action")]

    def get_msg_size(self):
        if self.comm_type == 0:
            return 0
        elif len(self.msg_type) == 0:
            return 1
        else:
            return sum([size for i,(size,_,__) in enumerate(self.get_msg_types()) if i in self.msg_type])
    
    def get_params(self):
        params = dict(**self.__dict__)
        params["patch_radius"] = self.patch.get_radius()
        params["s_init"] = self.patch.s_init
        params["msg_type"] = [self.get_msg_types()[type][2] for type in self.msg_type]
        params["comm_type"] = self.get_comm_types()[self.comm_type]
        del params["patch"]
        del params["agents"]
        del params["latest_obs"]
        del params["latest_patch"]
        return params

    def size(self):
        return self.x_max, self.y_max

    def get_states(self, agents_state, patch_state):
        # Initialize size of state
        obs_size = self.get_state_space()
        in_patch = np.array([self.agents[i].is_in_patch(agents_state[i], patch_state) for i in range(self.n_agents)])
        agents_obs = np.zeros((self.n_agents, obs_size))
        # Compute components of state
        for i in range(self.n_agents):
            agents_obs[i] = np.concatenate([agents_state[i], patch_state[:-1], [self.latest_patch]])
            if in_patch[i] or not self.in_patch_only:
                self.latest_patch = patch_state[-1]
        return agents_obs

    def get_state_space(self):
        obs_size = self.agents[0].num_vars + self.patch.num_vars
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
            # Update agent position, and send message
            agents_state[i, :agents_state.shape[1]] = self.agents[i].step(agents_state, actions)
            # Update agent energy
            agent_state, reward, s_eaten, penalty = self.agents[i].update_energy(agents_state[i], patch_state, action, dt=0.1)
            is_in_patch[i] = s_eaten != 0
            penalties[i,:] = penalty
            # Add agent reward to reward vector
            rewards[i] = reward
            tot_eaten += s_eaten
            agents_state[i] = agent_state
        # Compute welfare
        positive_rewards = rewards.copy()
        positive_rewards[positive_rewards < 0] = 0
        welfare = compute_NSW(positive_rewards) 
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
    def __init__(self,x,y,env,id=0):
        self.beta = env.beta # amount of eating per timestep
        self.v_max = env.v_max
        self.size = [env.x_max, env.y_max]
        self.e_init = env.e_init
        self.e_max = env.e_max
        self.id = id
        self.p_still = env.p_still
        self.p_act = env.p_act
        self.p_comm = env.p_comm
        self.p_att = env.p_att
        self.noise_rng = np.random.default_rng(seed=env.seed)
        self.damping = env.damping
        self.msg_size = env.get_msg_size()
        if len(env.msg_type) > 0:
            self.noise_arr = np.concatenate([[noise]*size for i,(size,noise,_) in enumerate(env.get_msg_types()) if i in env.msg_type])
        self.msg_type = env.msg_type
        self.comm_type = env.comm_type
        self.num_vars = 5+self.msg_size # Variables of interest: (x,y,v_x,v_y,e,[msg]) <- msg may not be there 
        
    def reset(self,x,y,rng):
        agent_state = np.zeros(self.num_vars)
        agent_state[:2] = np.array([x,y])
        vs = 2*self.v_max*rng.random(size=2) - self.v_max
        agent_state[2:4] = vs
        agent_state[4] = self.e_init
        return agent_state

    def dist_to(self,pos1,pos2):
        diff = pos1 - pos2
        return np.linalg.norm(diff, 2)
    
    def is_in_patch(self,agent_state,patch_state):
        dist = self.dist_to(agent_state[:2], patch_state[:2])
        return dist <= patch_state[2]

    """
    Message mechanism where agents are allowed to decide if they want to send/receive messages
    """
    def discrete_msg(self,agents_state,actions,msg):
        attention_other = actions[1-self.id][3]
        communication = actions[self.id][2]
        # When communicating send the message in full quality
        if attention_other > 0.5 and communication > 0.5:
            return msg
        # Add noise to message if not communicating
        noise = self.noise_rng.normal(0,self.noise_arr,size=self.noise_arr.shape)
        return msg + noise

    """
    Always send messages in full quality between agents
    """
    def always_msg(self,agents_state,actions,msg):
        return msg

    """
    Only sends messages with noise between agents
    """
    def never_msg(self,agents_state,actions,msg):
        return msg + self.noise_rng.normal(0,self.noise_arr,size=self.noise_arr.shape)

    def get_penalty(self,agent_state,patch_state,action):
        penalty = self.p_still
        if self.msg_size > 0:
            attention = action[3]
            communication = action[2]
            penalty += communication.item()*self.p_comm
            penalty += attention.item()*self.p_att
        max_penalty = np.linalg.norm(np.full_like(action[:2], self.v_max))
        penalty += np.linalg.norm(action[:2])/max_penalty*self.p_act
        return penalty
    
    def generate_message(self,agents_state,actions,msg):
        comm_types = [None, self.discrete_msg, self.always_msg, self.never_msg]
        return comm_types[self.comm_type](agents_state,actions,msg)
    
    def send_message(self,agents_state,actions):
        if self.msg_size == 0:
            return 0
        state = agents_state[self.id]
        msgs = [[state[4]], state[:2], state[2:4], actions[self.id][:2]]
        msg = np.concatenate([msgs[type] for type in self.msg_type])
        msg = self.generate_message(agents_state,actions,msg)
        if msg is not None:
            agents_state[1-self.id][5:] = msg
        
    
    def step(self,agents_state,actions):
        self.send_message(agents_state, actions)
        return self.update_position(agents_state[self.id],actions[self.id])
    
    def update_energy(self,agent_state,patch_state,action,dt=0.1):
        s_eaten = self.is_in_patch(agent_state, patch_state)*self.beta*patch_state[-1]
        penalty = self.get_penalty(agent_state, patch_state, action)
        # Update step (differential equation)
        de = s_eaten - dt*penalty
        agent_state[4] = agent_state[4]+de
        reward = agent_state[4]/self.e_init
        return agent_state, reward, s_eaten, penalty
        
    def update_position(self,agent_state,action, dt=0.1):
        # Functions needed to bound the allowed actions
        v_bounded = lambda v: np.clip(v, -self.v_max, self.v_max)
        # Compute action values
        pos = agent_state[:2].copy()
        vel = agent_state[2:4].copy()
        acc = action[:2]
        # Update position
        pos += dt*vel 
        pos = np.mod(pos, self.size)
        # Update velocity
        # TODO: Test velocity based control compared to acceleration based control
        # Velocity control
        #vel = v_bounded(acc - dt*(self.damping*acc))
        # Acceleration control
        vel = v_bounded(vel + dt*(acc - self.damping*vel))
        agent_state[:2] = pos
        agent_state[2:4] = vel
        return agent_state

class Patch:
    def __init__(self,x,y,radius,env):
        # Hyperparameters of dynamical system
        self.eta = env.eta # regeneration rate of resources
        self.gamma = env.env_gamma # decay rate of resources
        self.pos = np.array([x,y])
        self.radius = radius
        self.patch_resize = env.patch_resize
        self.s_init = env.s_init
        self.num_vars = 4 # Variables of interest: (x,y,r,s)
    
    def get_radius(self):
        return self.radius
        
    def update_resources(self, patch_state, eaten, dt=0.1):
        resources = patch_state[-1]
        scalars = np.array([dt*self.eta, -dt*self.gamma, -1])
        values = np.array([resources,np.power(resources,2),eaten])
        ds = np.dot(scalars.T, values)
        
        patch_state[-1] = np.clip(resources + ds, 0, self.s_init)
        return patch_state
        
    def reset(self):
        patch_state = np.zeros(self.num_vars)
        patch_state[:2] = self.pos
        patch_state[2] = self.radius
        patch_state[3] = self.s_init
        return patch_state