from td3 import *
import jax.numpy as jnp
import numpy as np
import jax
import gymnasium as gym
from flax import nnx
import matplotlib.pyplot as plt
from environment import *
from time import sleep

# TODO: Implement unit-tests for the Critic and Actor networks (below are some suggestions)
# 1. 

# Variables with tables for XOR problem
XOR_X = np.array([[1,1,1,0],
               [1,1,1,1],
               [1,1,0,0],
               [1,1,0,1],
               [1,0,1,0],
               [1,0,1,1],
               [1,0,0,0],
               [1,0,0,1],
               [0,1,1,0],
               [0,1,1,1],
               [0,1,0,0],
               [0,1,0,1],
               [0,0,1,0],
               [0,0,1,1],
               [0,0,0,0],
               [0,0,0,1]])
XOR_Y = np.array([[0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,1]]).T

def test_dimensions_actor():
    module = Actor(4, 1, 1, nnx.Rngs(0), hidden_dim=64)
    check_l1 = module.l1.kernel.value.shape == (4, 64)
    check_l2 = module.l2.kernel.value.shape == (64, 64)
    check_l3 = module.l3.kernel.value.shape == (64, 1)
    check = bool(check_l1 and check_l2 and check_l3)
    if not check:
        print("Error: Shape of the Actor network is not correct")
    return check

def test_target_shape():
    critic = Critic(2, nnx.Rngs(0), hidden_dim=64)
    actor = Actor(1, 1, 1, nnx.Rngs(0), hidden_dim=64)
    states = jnp.array([[1,2,3,4,5]]).T
    rs = jnp.array([[1,-1,0,0.5,0.1]]).T
    done = jnp.array([[1,0,0,0,0]]).T
    gamma = 0.99
    ys_pred = compute_targets(critic, actor, rs, states, done, gamma)
    ys_true = critic(states, actor(states))
    check = ys_pred.shape == ys_true.shape
    if not check:
        print("Error: Target doesn't have the same shape as the Critic's evaluation")
    return check
    

def test_dimensions_critic():
    module = Critic(4, nnx.Rngs(0), hidden_dim=64)
    check_l1 = module.l1.kernel.value.shape == (4, 64)
    check_l2 = module.l2.kernel.value.shape == (64, 64)
    check_l3 = module.l3.kernel.value.shape == (64, 1)
    check = bool(check_l1 and check_l2 and check_l3)
    if not check:
        print("Error: Shape of the Critic network is not correct")
    return check

def test_randomness_critic():
    module = Critic(4, nnx.Rngs(0), hidden_dim=64)
    l1_seq = np.array(module.l1.kernel.value.flatten())[:4]
    l2_seq = np.array(module.l2.kernel.value.flatten())[:4]
    l3_seq = np.array(module.l3.kernel.value.flatten())[:4]
    check = np.sum((l1_seq == l2_seq) | (l2_seq == l3_seq) | (l1_seq == l3_seq)) == 0
    if not check:
        print("Error: Critic parameters are not initialized randomly")
    return check
    

def test_randomness_actor():
    module = Actor(4, 1, 1, nnx.Rngs(0), hidden_dim=64)
    l1_seq = np.array(module.l1.kernel.value.flatten())[:4]
    l2_seq = np.array(module.l2.kernel.value.flatten())[:4]
    l3_seq = np.array(module.l3.kernel.value.flatten())[:4]
    check = np.sum((l1_seq == l2_seq) | (l2_seq == l3_seq) | (l1_seq == l3_seq)) == 0
    if not check:
        print("Error: Actor parameters are not initialized randomly")
    return check
    
def test_output_bound_actor():
    module = Actor(4, 1, 1, nnx.Rngs(0), hidden_dim=64)
    inputs = jax.random.normal(jax.random.key(0), shape=(10,4))
    outputs = module(inputs)
    check = bool(jnp.all((outputs <= 1) & (outputs >= -1)))
    if not check:
        print("Error: Bound violation on the Actor output")
    return check

def test_gradients_actor():
    actor = Actor(4,1,1, nnx.Rngs(0), hidden_dim=64)
    critic = Critic(5, nnx.Rngs(0), hidden_dim=64)
    
    optimizer = nnx.Optimizer(actor, optax.adam(1e-3))
    states = jax.random.normal(jax.random.key(0), shape=(10,4))
    loss, grads = mean_optimize_actor(optimizer, actor, critic, states)
    summed_value = grads.l1.kernel.value.sum() + grads.l2.kernel.value.sum() + grads.l3.kernel.value.sum()
    if summed_value == 0:
        print("Error: Gradients' sum of Actor optimization is zero")
    return bool(summed_value != 0)

def test_gradients_critic():
    critic = Critic(5, nnx.Rngs(0), hidden_dim=64)
    optimizer = nnx.Optimizer(critic, optax.adam(1e-3))
    states = jax.random.normal(jax.random.key(0), shape=(10,4))
    actions = jax.random.normal(jax.random.key(0), shape=(10,1))
    ys = jax.random.normal(jax.random.key(0), shape=(10,1))
    loss, grads = MSE_optimize_critic(optimizer, critic, states, actions, ys)
    summed_value = grads.l1.kernel.value.sum() + grads.l2.kernel.value.sum() + grads.l3.kernel.value.sum()
    if summed_value == 0:
        print("Error: Gradients' sum of Critic optimization is zero")
    return bool(summed_value != 0)

def test_XOR_benchmark_critic():
    np.random.seed(0)
    states = XOR_X
    actions = np.zeros((16,1))
    ys = XOR_Y
    p_0 = 1/24
    p_1 = 1/8
    ps = [p_0, p_0, p_0, p_0, p_0, p_0, p_1, p_0, p_0, p_0, p_1, p_0, p_1, p_0, p_0, p_1]
    critic = Critic(5, nnx.Rngs(0), hidden_dim=256)
    optimizer = nnx.Optimizer(critic, optax.adam(2e-3))
    max_it = 200
    batch_size = 16
    for i in range(max_it):
        selection = np.random.choice(np.arange(0,states.shape[0],1), p=ps, size=(batch_size,))
        states_select = jnp.array(states[selection, :])
        ys_select = jnp.array(ys[selection, :])
        actions_select = jnp.array(actions[selection, :])
        loss, grads = MSE_optimize_critic(optimizer, critic, states_select, actions_select, ys_select)
        check = loss < 1e-3
        if check:
            break
    if not check:
        print("Error: Critic not able to solve XOR problem")
    return check

def test_XOR_benchmark_actor():
    np.random.seed(0)
    states = XOR_X
    ys = XOR_Y
    p_0 = 1/24
    p_1 = 1/8
    ps = [p_0, p_0, p_0, p_0, p_0, p_0, p_1, p_0, p_0, p_0, p_1, p_0, p_1, p_0, p_0, p_1]
    actor = Actor(4, 1, 1, nnx.Rngs(0), hidden_dim=256)
    optimizer = nnx.Optimizer(actor, optax.adam(2e-3))
    max_it = 200
    batch_size = 16
    for i in range(max_it):
        selection = np.random.choice(np.arange(0,states.shape[0],1), p=ps, size=(batch_size,))
        states_select = jnp.array(states[selection, :])
        ys_select = jnp.array(ys[selection, :])
        loss_fn = lambda actor: ((actor(states_select) - ys_select) ** 2).mean() # MSE loss
        loss, grads = nnx.value_and_grad(loss_fn)(actor)
        optimizer.update(grads)
        check = loss < 1e-3
        if check:
            break
    
    if not check:
        print("Error: Actor not able to solve XOR problem")
    return check

def test_ptr_buffer():
    buffer = Buffer(10, 1, 1)
    check1 = buffer.ptr == 0
    buffer.add(0,0,0,0,0)
    check2 = buffer.ptr == 1
    if not (check1 and check2):
        print("Error: Buffer pointer is not updating")
    return (check1 and check2)

def test_randomness_buffer():
    buffer = Buffer(10, 1, 1)
    for i in range(9):
        buffer.add(i,i,i,i,i)
    state1, *_ = buffer.sample(100)
    state2, *_ = buffer.sample(100)
    check = np.sum(state1 == state2) < 20
    if not check:
        print("Error: Buffer not sampling randomly")
    return check
    

def test_shape_buffer_samples():
    buffer = Buffer(10, 1, 1)
    check = buffer.states.shape == (10,1)
    if not check:
        print("Error: Buffer element shape not correct, should be (buffer_size, 1)")
    return check

def test_overflowing_buffer():
    buffer = Buffer(10, 1, 1)
    for i in range(10):
        buffer.add(0,0,0,0,0)
    check = buffer.ptr == 0 and buffer.size == buffer.max_size
    if not check:
        print("Error: Buffer overflow behavior is incorrect")
    return check

def test_sample_action():
    a_max = 0.5
    actor = Actor(1,1,a_max, nnx.Rngs(0))
    state = jnp.array([1])
    action_before = actor(state)
    key = jax.random.PRNGKey(0)
    for i in range(50):
        key,subkey = jax.random.split(key)
        action_after = sample_action(key, actor, state, -a_max, a_max)
        check = action_after <= a_max and action_after >= -a_max and action_after != action_before
        if not check:
            print("Error: Actions are not sampled randomly from Actor")
            break
    return check

def test_action_execution():
    rng = jax.random.PRNGKey(62)
    reset_key, step_key1, step_key2, step_key3 = jax.random.split(rng,4)
    env, env_params = gymnax.make("Pendulum-v1")
    # Left (Clockwise) torque
    obs, state = env.reset(reset_key, env_params)
    obs_next, state_next, reward, done, _ = env.step(step_key1,state,-2,env_params)
    check1 = (obs_next[2] - obs[2]) < 0
    # Right (Anti-clockwise) torque
    obs, state = env.reset(reset_key, env_params)
    obs_next, state_next, reward, done, _ = env.step(step_key2,state,2,env_params)
    check2 = (obs_next[2] - obs[2]) > 0
    # No torque (But gravity is expected)
    obs, state = env.reset(reset_key, env_params)
    obs_next, state_next, reward, done, _ = env.step(step_key3,state,0,env_params)
    check3 = (obs[0] > 0 and (obs_next[2] - obs[2]) < 0) or (obs[0] < 0 and (obs_next[2] - obs[2]) > 0)
    check = check1 and check2 and check3
    if not check:
        print("Error: Unexpected env.step() behavior in gymnax Pendulum-v1 environment")
    return check

def test_polyak_update():
    a1 = Actor(1,1,1,nnx.Rngs(0),hidden_dim=6)
    a2 = Actor(1,1,1,nnx.Rngs(1),hidden_dim=6)
    a1_state = nnx.state(a1)
    a2_state= nnx.state(a2)
    tau = 0.6
    a1_new = polyak_update(tau, a1, a2)
    a1_manual = a2_state.l1.kernel.value*tau + a1_state.l1.kernel.value*(1-tau)
    filter = a1_new.l1.kernel.value - a1_manual < 0.0000001 # Apparently JAX sometimes has very little numerical instabilities
    check = jnp.all(filter)
    if not check:
        print("Error: Polyak update not updating parameters correctly")
    return check

def test_ddpg_train_pendulum():
    episodes = list(range(1,5))
    rewards, actor, critic, reset_key = train_ddpg(gym.make("Pendulum-v1"), episodes[-1], hidden_dim=32)
    input("Press enter to see reward plot...")
    plt.figure()
    plt.plot(episodes, rewards)
    plt.show()
    input("Press enter to see trained model in action...")
    env = gym.make("Pendulum-v1", render_mode="human")
    state, info = env.reset(seed=reset_key)
    while True:
        state, reward, terminated, truncated, _ = env.step(actor(state))
        if terminated or truncated:
            break
    return True

def test_ddpg_train_patch():
    episodes = list(range(1,11))
    env = Environment(patch_radius=0.5, step_max=400, alpha=2)
    action_dim, a_range = env.get_action_space()
    rewards, actor, critic, reset_key = train_ddpg(env, episodes[-1], lr_c=1e-3, lr_a=3e-4, tau=0.1, action_dim=action_dim, state_dim=env.get_state_space()[1], action_max=a_range[1], hidden_dim=32, batch_size=100, seed=0)
    input("Press enter to see trained model in action...")
    env = RenderEnvironment(env)
    state, info = env.reset(seed=reset_key)
    while True:
        state, reward, terminated, truncated, _ = env.step(actor(state))
        if terminated or truncated:
            break
    env.render()
    return True

if __name__ == "__main__":
    result = []
    # result.append(test_dimensions_actor())
    # result.append(test_dimensions_critic())
    # result.append(test_output_bound_actor())
    # result.append(test_gradients_actor())
    # result.append(test_gradients_critic())
    # result.append(test_XOR_benchmark_critic())
    # result.append(test_XOR_benchmark_actor())
    # result.append(test_randomness_critic())
    # result.append(test_randomness_actor())
    # result.append(test_target_shape())
    # result.append(test_overflowing_buffer())
    # result.append(test_ptr_buffer())
    # result.append(test_randomness_buffer())
    # result.append(test_shape_buffer_samples())
    # result.append(test_sample_action())
    # result.append(test_action_execution())
    # result.append(test_polyak_update())
    result.append(test_ddpg_train_patch())
    if len(result) == sum(result):
        print("Success: All tests have passed")

