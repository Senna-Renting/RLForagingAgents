import jax.numpy as jnp
from typing import NamedTuple, List, Dict
from functools import partial
import jax
# TODO: Rewrite the environment in pure functions here to be handled by jax
class TrainParameters(NamedTuple):
    current_path: str
    hidden_dims: tuple
    action_range: tuple
    batch_size: jnp.integer
    num_episodes: jnp.integer
    seed: jnp.integer
    step_max: jnp.integer
    tau: jnp.float32
    gamma: jnp.float32
    act_noise: jnp.float32
    lr_a: jnp.float32
    lr_c: jnp.float32
    
    

class EnvParameters(NamedTuple):
    # General
    env_size: jnp.float32
    p_welfare: jnp.float32
    dt: jnp.float32
    # Communication
    p_att: jnp.float32
    p_comm: jnp.float32
    msg_noise: jnp.float32
    comm_type: jnp.integer
    # Patch
    growth: jnp.float32
    decay: jnp.float32
    patch_resource_init: jnp.float32
    patch_radius: jnp.float32
    # Agents
    p_still: jnp.float32
    p_act: jnp.float32
    n_agents: jnp.integer
    v_max: jnp.float32
    e_max: jnp.float32
    eat_rate: jnp.float32
    e_init: jnp.float32
    damping: jnp.float32

class AgentState(NamedTuple):
    position: jnp.ndarray
    velocity: jnp.ndarray
    energy: jnp.ndarray

class PatchState(NamedTuple):
    position: jnp.ndarray
    radius: jnp.ndarray
    resource: jnp.ndarray

class EnvState(NamedTuple):
    patch_state: PatchState
    agent_states: List[AgentState]

"""
Reset the environment to an initial random state
"""
@partial(jax.jit, static_argnames=["parameters"])
def env_reset(key: jax.random.PRNGKey, parameters: EnvParameters):
    key1, key2, key3 = jax.random.split(key, 3)
    n_agents = parameters.n_agents
    v_max = parameters.v_max
    size = parameters.env_size
    patch_position = jnp.array([size/2,size/2], dtype=jnp.float32)
    patch_state = PatchState(position=patch_position, 
                             radius=jnp.array([parameters.patch_radius]),
                             resource=jnp.array([parameters.patch_resource_init]))
    positions = jax.random.uniform(key1, shape=(n_agents,2), minval=0, maxval=parameters.env_size)
    velocities = jax.random.uniform(key2, shape=(n_agents,2), minval=-v_max, maxval=v_max)
    agent_states = [AgentState(position=positions.at[i_a].get(),
                               velocity=velocities.at[i_a].get(),
                               energy=jnp.array([parameters.e_init])
                    ) for i_a in range(n_agents)]
    no_actions = jnp.zeros((4,n_agents))
    messages = get_messages(agent_states, no_actions, key3, parameters)
    agents_obs = get_obs(agent_states, patch_state, messages, parameters)
    return EnvState(patch_state=patch_state, agent_states=agent_states), agents_obs

"""
Computes a single step through the environment
"""
@partial(jax.jit, static_argnames=["parameters"])
def env_step(state: EnvState, actions: jnp.ndarray, key: jax.random.PRNGKey, parameters: EnvParameters):
    patch_state = state.patch_state
    agent_states = state.agent_states
    n_agents = parameters.n_agents
    p_welfare = parameters.p_welfare
    tot_eaten = 0
    rewards = jnp.zeros((n_agents,1))
    for i_a in range(n_agents):
        action = actions.at[i_a].get()
        agent_state = ag_position_step(agent_states[i_a], action, parameters)
        agent_state, reward, s_eaten, penalty = ag_energy_step(patch_state, agent_state, action, parameters)
        rewards = rewards.at[i_a].set(reward)
        tot_eaten += s_eaten
        agent_states[i_a] = agent_state
    
    # Modify reward with welfare contribution
    positive_rewards = jnp.maximum(rewards, jnp.zeros(n_agents))
    welfare = jnp.power(jnp.prod(positive_rewards), 1/rewards.shape[0])
    rewards = (1-p_welfare)*rewards + p_welfare*welfare
    
    patch_state = patch_step(patch_state, tot_eaten, parameters)
    messages = get_messages(agent_states, actions, key, parameters)
    agents_obs = get_obs(agent_states, patch_state, messages, parameters)
    done = False
    return EnvState(patch_state=patch_state, agent_states=agent_states), agents_obs, rewards, done

def ag_position_step(agent_state: AgentState, action: jnp.ndarray, parameters: EnvParameters):
    # Compute action values
    pos = agent_state.position
    vel = agent_state.velocity
    dt = parameters.dt
    size = parameters.env_size
    damping = parameters.damping
    vmax = parameters.v_max
    acc = action[:2]
    # Update position
    pos = jnp.mod(pos + dt*vel, size) 
    # Update velocity
    # TODO: Test velocity based control compared to acceleration based control
    # Velocity control
    #vel = v_bounded(acc - dt*(self.damping*acc))
    # Acceleration control
    vel = jnp.clip(vel + dt*(acc - damping*vel), -vmax, vmax)
    return AgentState(position=pos, velocity=vel, energy=agent_state.energy)

def patch_step(patch_state: PatchState, eaten: jnp.float32, parameters: EnvParameters):
    resource = patch_state.resource
    dt = parameters.dt
    growth = parameters.growth
    decay = parameters.decay
    s_init = parameters.patch_resource_init
    scalars = jnp.array([dt*growth, -dt*decay, -1])
    values = jnp.array([resource,jnp.power(resource,2),eaten])
    ds = jnp.dot(scalars.T, values)    
    resource = jnp.clip(resource + ds, 0, s_init)
    patch_state = PatchState(position=patch_state.position, 
                             radius=patch_state.radius,
                             resource=resource)
    return patch_state

def ag_energy_step(patch_state: PatchState, agent_state: AgentState, action: jnp.ndarray, parameters: EnvParameters):
    eat_rate = parameters.eat_rate
    dt = parameters.dt
    e_init = parameters.e_init
    resource = patch_state.resource
    energy = agent_state.energy
    s_eaten = ag_is_in_patch(patch_state, agent_state)*eat_rate*resource
    penalty = get_penalty(agent_state, action, parameters)
    # Update step (differential equation)
    de = s_eaten - dt*penalty
    energy = energy+de
    reward = energy/e_init
    agent_state = AgentState(position=agent_state.position, 
                             velocity=agent_state.velocity,
                             energy=energy)
    return agent_state, reward, s_eaten, penalty

def ag_is_in_patch(patch_state: PatchState, agent_state: AgentState):
    def dist_to(pos1,pos2):
        diff = pos1 - pos2
        return jnp.linalg.norm(diff, 2)
    ag_pos = agent_state.position
    p_pos = patch_state.position
    p_radius = patch_state.radius
    dist = dist_to(ag_pos, p_pos)
    return dist <= p_radius

def get_penalty(agent_state: AgentState, action: jnp.ndarray, parameters: EnvParameters):
    p_still = parameters.p_still
    p_act = parameters.p_act
    p_att = parameters.p_att
    p_comm = parameters.p_comm
    v_max = parameters.v_max
    penalty = p_still    
    attention = action[3]
    communication = action[2]
    penalty += communication*p_comm
    penalty += attention*p_att
    max_penalty = jnp.linalg.norm(jnp.full_like(action[:2], v_max))
    penalty += jnp.linalg.norm(action[:2])/max_penalty*p_act
    return penalty

def get_messages(agent_states: List[AgentState], actions: jnp.ndarray, key: jax.random.PRNGKey, parameters: EnvParameters):
    msgs = jnp.array([jnp.concatenate(jax.tree.leaves(agent_state)) for i_a, agent_state in enumerate(reversed(agent_states))])
    max_vals = jnp.array([jnp.concatenate([jnp.full_like(a_state.position, parameters.env_size),
                jnp.full_like(a_state.velocity, parameters.v_max),
                jnp.full_like(a_state.energy, parameters.e_max)]) for a_state in agent_states])
    noise_lvl = parameters.msg_noise*max_vals
    noise = noise_lvl*jax.random.normal(key, msgs.shape)
    # No message state
    if parameters.comm_type == 0:
        return jnp.array([[],[]])
    # Learn communication
    if parameters.comm_type == 1:
        for i_a in range(parameters.n_agents):
            noisy = lambda i_a, msgs, noise: msgs.at[i_a].set(msgs.at[i_a].get() + noise.at[i_a].get())
            no_noisy = lambda i_a, msgs, noise: msgs
            att_other = actions[1-i_a,3]
            comm = actions[i_a,2]
            return jax.lax.cond((att_other > 0.5) & (comm > 0.5), no_noisy, noisy, i_a, msgs, noise)
    # Message with noise always
    if parameters.comm_type == 2:
        return msgs + noise
    # Message without noise always
    if parameters.comm_type == 3:
        return msgs
    # Default is message without noise
    return msgs

# TODO: fully implement this observation method to behave like our numpy environment
def get_obs(agent_states: List[AgentState], patch_state: PatchState, messages: jnp.ndarray, parameters: EnvParameters):
    n_agents = parameters.n_agents
    agents_obs = jnp.array([jnp.concatenate(jax.tree.leaves([agent_state, patch_state, messages.at[i_a].get()])) 
     for i_a, agent_state in enumerate(agent_states)])
    return agents_obs

def get_action_space(parameters: EnvParameters):
    return 2*(1+(parameters.comm_type>0))

if __name__ == "__main__":
    parameters = EnvParameters(
        env_size = 50,
        p_welfare = 0.9,
        dt = 0.1,
        p_att = 0.02,
        p_comm = 0.1,
        msg_noise = 0.1,
        comm_type = 1,
        growth = 0.1,
        decay = 0.01,
        patch_resource_init = 10,
        patch_radius = 10,
        p_still = 0.02,
        p_act = 0.2,
        n_agents = 2,
        v_max = 4,
        e_max = 10 + 5,
        eat_rate = 0.1,
        e_init = 5,
        damping = 0.3
    )
    key = jax.random.PRNGKey(0)
    env, agents_obs = env_reset(key, parameters)
    print("Obs i=0: ", agents_obs)
    action = jnp.array([[1,1,0,0], [-1,-1.5,0,0]])
    sum = 0
    for i in range(10023):
        if i % 2 == 0:
            env, agents_obs, rewards, done = env_step(env, action, key, parameters)
        sum += 1
        if i < 10:
            print("Rewards: ", rewards)
    print(f"Obs: i={i}: ", agents_obs)
    print("Rewards: ", rewards)
    


    
    