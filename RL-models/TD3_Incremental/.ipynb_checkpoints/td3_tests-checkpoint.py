from td3 import *
import jax.numpy as jnp
import numpy as np
import jax
from flax import nnx

# TODO: Implement unit-tests for the Critic and Actor networks (below are some suggestions)
# 1. Is the shape of the targets equal to the output shape of the critic network?

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

if __name__ == "__main__":
    result = []
    result.append(test_dimensions_actor())
    result.append(test_dimensions_critic())
    result.append(test_output_bound_actor())
    result.append(test_gradients_actor())
    result.append(test_gradients_critic())
    result.append(test_XOR_benchmark_critic())
    result.append(test_XOR_benchmark_actor())
    result.append(test_randomness_critic())
    result.append(test_randomness_actor())
    result.append(test_target_shape())
    if len(result) == sum(result):
        print("Success: All tests have passed")

