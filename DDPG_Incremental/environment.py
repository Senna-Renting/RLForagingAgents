import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os
from abc import ABC, abstractmethod
from functools import partial

# Helper function for computing the Nash Social Welfare function (aka geometric mean)
def compute_NSW(rewards):
    NSW = np.power(np.prod(rewards), 1/rewards.shape[0])
    return NSW
    
class NAgentsEnv():
    def __init__(self, patch_radius=10,s_init=10, e_init=5, eta=0.1, beta=0.05, env_gamma=0.01, step_max=600, x_max=50, y_max=50, v_max=4, n_agents=2, in_patch_only=False, p_welfare=0, rof=0, patch_resize=False, agent_type="No Communication", **kwargs):
        self.x_max = x_max
        self.y_max = y_max
        self.v_max = v_max
        # Eta and gamma are used for computing the rendering (don't remove them!!)
        self.eta = eta
        self.env_gamma = env_gamma
        self.step_max = step_max
        self.step_idx = 0
        self.rof = rof
        self.patch = Patch(x_max/2, y_max/2, patch_radius, s_init, eta=eta, gamma=env_gamma, rof=rof, patch_resize=patch_resize)
        self.damping = 0.3
        self.p_still = 0.02
        self.p_act = 0.2
        self.p_att = 0.02
        self.p_comm = 0.1
        self.p_rof = 0.2
        self.agent_type = agent_type
        if self.agent_type == "Learned Communication":
            self.agents = [LearnedCommsAgent(0,0,x_max,y_max,e_init,v_max, beta=beta, id=i, damping=self.damping, p_still=self.p_still, p_act=self.p_act, p_att=self.p_att, p_comm=self.p_comm, p_rof=self.p_rof, seed=kwargs["seed"]) for i in range(n_agents)]
        elif self.agent_type == "State Communication":
            self.agents = [StateCommsAgent(0,0,x_max,y_max,e_init,v_max, beta=beta, id=i, damping=self.damping, p_still=self.p_still, p_act=self.p_act, p_att=self.p_att, p_rof=self.p_rof, seed=kwargs["seed"]) for i in range(n_agents)]
        elif self.agent_type == "SA-State Communication":
            self.agents = [SendAcceptStateCommsAgent(0,0,x_max,y_max,e_init,v_max, beta=beta, id=i, damping=self.damping, p_still=self.p_still, p_act=self.p_act, p_att=self.p_att, p_comm=self.p_comm, p_rof=self.p_rof, seed=kwargs["seed"]) for i in range(n_agents)]
        else:
            self.agents = [Agent(0,0,x_max,y_max,e_init,v_max, beta=beta, id=i, damping=self.damping, p_still=self.p_still, p_act=self.p_act, p_rof=self.p_rof, seed=seed) for i in range(n_agents)]
        self.n_agents = n_agents
        self.beta = beta
        self.e_init = e_init
        self.in_patch_only = in_patch_only
        self.patch_resize = patch_resize
        self.p_welfare = p_welfare
        # Initialize latest observation states
        self.latest_obs = np.zeros((self.n_agents, self.n_agents, self.agents[0].num_vars-1))
        self.latest_patch = 0

    def get_action_space(self):
        message_dim = 0
        if self.agent_type == "Learned Communication":
            message_dim = 2
        if self.agent_type == "SA-State Communication":
            message_dim = 5
        if self.agent_type == "State Communication":
            message_dim = 1
        action_dim = 2 # We use two acceleration terms (one for x and one for y)
        action_range = [-self.v_max, self.v_max]
        print(action_dim+message_dim)
        return action_dim+message_dim, action_range
    
    def get_params(self):
        params = dict(**self.__dict__)
        params["patch_radius"] = self.patch.get_radius()
        params["s_init"] = self.patch.s_init
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
            # Update agent position
            a_state = agents_state[i]
            agents_state[i, :agents_state.shape[1]] = self.agents[i].update_position(a_state, action)
            if self.agent_type in ["State Communication", "SA-State Communication"]:    
                self.agents[i].get_message(agents_state, actions)
            if self.agent_type == "Learned Communication":
                msg = self.agents[i].update_message(agents_state, action)
                agents_state[1-i, -1] = msg
            # Update agent energy
            agent_state, reward, s_eaten, penalty = self.agents[i].update_energy(agents_state, patch_state, action, dt=0.1)
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
    def __init__(self,x,y,x_max,y_max,e_init,v_max,alpha=0.4, beta=0.25, id=0, p_still=0.2, p_act=0.8, p_rof=0.2, damping=0.3, seed=0):
        # Hyperparameters of dynamical system
        self.beta = beta # amount of eating per timestep
        # General variables
        self.v_max = v_max
        self.size = [x_max, y_max]
        self.e_init = e_init
        self.id = id
        self.p_still = p_still
        self.p_act = p_act
        self.p_rof = p_rof
        self.damping = damping
        self.num_vars = 5 # Variables of interest: (x,y,v_x,v_y,e)
    
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
        return dist <= patch_state[3]

    def is_in_rof(self,agent_state,patch_state):
        dist = self.dist_to(agent_state[:2], patch_state[:2])
        return dist > patch_state[3] and dist <= (patch_state[3] + patch_state[2])

    def update_energy(self,agents_state,patch_state,action,other_penalties=0,dt=0.1):
        # Amount eaten depends on other agents in patch
        agent_state = agents_state[self.id]
        who_in = [self.is_in_patch(agents_state[i], patch_state) for i in range(len(agents_state))]
        s_eaten = who_in[self.id]*self.beta*patch_state[4]
        if np.any(who_in):
            s_eaten /= np.sum(who_in) 
        max_penalty = np.linalg.norm(np.full(2, self.v_max))
        action_p = np.linalg.norm(action[:2])/max_penalty*self.p_act
        rof_p = (self.is_in_rof(agent_state,patch_state)).astype(int)*self.p_rof
        # Update step (differential equation)
        de = s_eaten - dt*(action_p + rof_p + other_penalties)
        # If agent has negative or zero energy, put the energy value at zero and consider the agent dead
        # Also agent can't have more energy than it's initial energy value
        agent_state[4] = agent_state[4]+de
        reward = agent_state[4]/self.e_init
        penalties = action_p
        #print("Regular pens: ", action_p + rof_p, "\n Other pens: ", other_penalties)
        return agent_state, reward, s_eaten, penalties
        
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
        vel = v_bounded(vel + dt*(acc-self.damping*vel))
        agent_state[:2] = pos
        agent_state[2:4] = vel
        return agent_state

class LearnedCommsAgent(Agent):
    def __init__(self,x,y,x_max,y_max,e_init,v_max,alpha=0.4, beta=0.25, id=0, p_still=0.05, p_act=0.3, p_comm=0.4, p_att=0.4, p_rof=0.3, damping=0.3, seed=0):
        super().__init__(x,y,x_max,y_max,e_init,v_max,alpha,beta,id,p_still,p_act,p_rof,damping,seed)
        self.p_comm = p_comm
        self.p_att = p_att
        self.noise_rng = np.random.default_rng(seed=seed)
        self.num_vars = 6
    
    # We will add the message as a last term to the agent's state
    # For now only works with 2 agents
    def update_message(self,agents_state,action):
        msg = action[2]
        attention = (self.v_max+action[3])/2*self.v_max # Bounds attention to [0,1]
        agents_pos = [a[:2] for a in agents_state]
        max_dist = np.sqrt(self.size[0]**2+self.size[1]**2)
        noise_lvl = self.dist_to(*agents_pos)*(self.v_max-attention)/(max_dist*self.v_max) # Bounds noise lvl to [0,1]
        noise_value = self.noise_rng.normal(0,noise_lvl)
        noised_msg = np.clip(msg + noise_value, -self.v_max, self.v_max)
        return noised_msg # Test the message itself first before adding noise
    
    def update_energy(self,agents_state,patch_state,action,other_penalties=0,dt=0.1):
        penalty = np.abs(action[2]/self.v_max)*self.p_comm
        penalty += (action[3]+self.v_max)/(2*self.v_max)*self.p_att
        return super().update_energy(agents_state,patch_state,action,other_penalties=penalty,dt=0.1)

class StateCommsAgent(Agent):
    def __init__(self,x,y,x_max,y_max,e_init,v_max,alpha=0.4, beta=0.25, id=0, p_still=0.05, p_act=0.3, p_att=0.4, p_rof=0.3, damping=0.3, seed=0,):
        super().__init__(x,y,x_max,y_max,e_init,v_max,alpha,beta,id,p_still,p_act,p_rof,damping,seed)
        self.p_att = p_att
        self.noise_rng = np.random.default_rng(seed=seed)
        self.num_vars += 5

    def get_message(self,agents_state,actions):
        msg = agents_state[1-self.id][:5]
        attention = (self.v_max+action[2])/(2*self.v_max) # Bounds attention to [0,1]
        agents_pos = [a[:2] for a in agents_state]
        max_dist = np.sqrt(self.size[0]**2+self.size[1]**2)
        noise_lvl = self.dist_to(*agents_pos)/max_dist # Bounds noise lvl to [0,1]
        noise_values = self.noise_rng.normal(0,noise_lvl,size=msg.shape)
        if round(attention) == 1:
            return  msg + noise_values # Test the message itself first before adding noise
        else:
            return np.zeros(msg.shape) # Zero vector indicating no information is retrieved

    def update_energy(self,agents_state,patch_state,action,other_penalties=0,dt=0.1):
        penalty = (action[-1]+self.v_max)/(2*self.v_max)*self.p_att
        return super().update_energy(agents_state,patch_state,action,other_penalties=penalty,dt=0.1)

class SendAcceptStateCommsAgent(StateCommsAgent):
    def __init__(self,x,y,x_max,y_max,e_init,v_max,alpha=0.4, beta=0.25, id=0, p_still=0.05, p_act=0.3, p_att=0.4, p_comm=0.4, p_rof=0.3, damping=0.3, seed=0, signals=[0,1,2,3]):
        self.p_comm = p_comm
        super().__init__(x,y,x_max,y_max,e_init,v_max,alpha,beta,id,p_still,p_act,p_att,p_rof,damping,seed)
        self.num_vars = 7 # 5 state dimensions and 2 message dimensions
        
    def get_message(self,agents_state,actions):
        state_other = agents_state[1-self.id][:5]
        # Message can be position, velocity, energy or action of other agent respectively
        msg = np.array([state_other[:2], state_other[2:4], [state_other[4],0], actions[1-self.id][:2]])
        normalize = lambda x: (self.v_max+x)/(2*self.v_max) # Bounds action value to [0,1]
        attention = normalize(actions[self.id][-1])
        # Communication here becomes a one-hot encoded signal that is selected by taking the strongest component
        communicate = normalize(actions[1-self.id][2:-1])
        value = np.max(communicate)
        idx = np.argmax(communicate)
        msg = np.dot(communicate, msg)
        agents_pos = [a[:2] for a in agents_state]
        max_dist = np.sqrt(self.size[0]**2+self.size[1]**2)
        noise_lvl = (1-value)*(1+(1-attention)*(self.dist_to(*agents_pos)/max_dist)) # std 1 noise max
        noise_values = self.noise_rng.normal(0,noise_lvl,size=1)
        agents_state[self.id][5:] = msg + noise_values # Test the message itself first before adding noise

    def update_energy(self,agents_state,patch_state,action,other_penalties=0,dt=0.1):
        a_comm = action[2:-1]
        comm_p = np.linalg.norm(a_comm)/np.linalg.norm(np.full_like(a_comm, self.v_max))*self.p_comm
        att_p = (action[-1]+self.v_max)/(2*self.v_max)*self.p_att
        # Amount eaten depends on other agents in patch
        agent_state = agents_state[self.id]
        who_in = [self.is_in_patch(agents_state[i], patch_state) for i in range(len(agents_state))]
        s_eaten = who_in[self.id]*self.beta*patch_state[4]
        if np.any(who_in):
            s_eaten /= np.sum(who_in) 
        max_penalty = np.linalg.norm(np.full(2, self.v_max))
        action_p = np.linalg.norm(action[:2])/max_penalty*self.p_act
        rof_p = (self.is_in_rof(agent_state,patch_state)).astype(int)*self.p_rof
        # Update step (differential equation)
        de = s_eaten - dt*(action_p + rof_p + comm_p + att_p)
        # If agent has negative or zero energy, put the energy value at zero and consider the agent dead
        # Also agent can't have more energy than it's initial energy value
        agent_state[4] = agent_state[4]+de
        reward = agent_state[4]/self.e_init
        penalties = action_p
        return agent_state, reward, s_eaten, penalties

class Patch:
    def __init__(self, x,y,radius,s_init, eta=0.1, gamma=0.01, rof=0, patch_resize=False):
        # Hyperparameters of dynamical system
        self.eta = eta # regeneration rate of resources
        self.gamma = gamma # decay rate of resources
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
        scalars = np.array([dt*self.eta, -dt*self.gamma, -1])
        values = np.array([resources,np.power(resources,2),eaten])
        ds = np.dot(scalars.T, values)
        
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