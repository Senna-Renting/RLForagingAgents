from environment import *
import time

def test_env_initialization():
    env_state = initialize_env(2, True, zero_sw, 0)
    env_state = initialize_patch(env_state)
    env_state = initialize_agents(jax.random.PRNGKey(0), env_state)
    print(env_state.get("agents_state").get("position"), env_state.get("patch_state").get("position"))
    print(get_env_obs(env_state))
    return True

def test_in_patch():
    env_state = initialize_env(2, True, zero_sw, 0)
    env_state = initialize_patch(env_state)
    print(is_in_patch(jnp.array([[2.1,2.5],[2.0,2.5]]), env_state))
    return True

def test_update_energy():
    env_state = initialize_env(2, True, zero_sw, 0)
    env_state = initialize_patch(env_state)
    env_state = initialize_agents(jax.random.PRNGKey(0), env_state)
    actions = jnp.array([[0.1,-0.1],[0.01,-0.2]])
    print(env_state)
    env_state = update_energy(actions, env_state)
    print(env_state)
    return True

def test_update_position():
    env_state = initialize_env(2, True, zero_sw, 0)
    env_state = initialize_patch(env_state)
    env_state = initialize_agents(jax.random.PRNGKey(0), env_state)
    actions = jnp.array([[0.1,-0.1],[0.01,-0.2]])
    print(env_state)
    env_state = update_position(actions, env_state)
    print(env_state)
    return True

def test_update_resources():
    env_state = initialize_env(2, True, zero_sw, 0)
    env_state = initialize_patch(env_state)
    env_state = initialize_agents(jax.random.PRNGKey(0), env_state)
    env_state = copy(env_state, {"agents_state": copy(env_state.get("agents_state"), {"tot_eaten": 0.5})})
    print(env_state)
    env_state = update_resources(env_state)
    print(env_state)
    return True

def compare_envs():
    start_time = time.time()
    env_state = initialize_env(2, True, zero_sw, 0)
    env = NAgentsEnv(n_agents=2, obs_others=True, seed=0, sw_fun=zero_sw)
    env_state2, _ = env.reset(seed=0)
    env_state, states = reset_env(jax.random.PRNGKey(0), env_state)
    actions = jnp.array([[0.1,0.2],[-0.2,0.]])
    step = lambda i,env_s: step_env(actions, env_s)[0]
    start_time = time.time()
    #env_state = jax.lax.fori_loop(1,400,step,env_state)
    for i in range(400):
        env_state = step(i,env_state)
    end_time = time.time()
    print(f"Time needed for one episode run jax FrozenDict: {end_time - start_time}")
    start_time2 = time.time()
    #env_state = jax.lax.fori_loop(1,400,step,env_state)
    for i in range(400):
        print(i)
        env_state2, *_ = env.step(env_state2, *[jnp.array([0.1,0.2]), jnp.array([-0.2,0])])
    end_time2 = time.time()
    print(f"Time needed for one episode run numpy: {end_time2 - start_time2}")
    return True

if __name__ == "__main__":
    result = []
    # add all the tests below
    #result.append(test_env_initialization())
    #result.append(test_in_patch())
    #result.append(test_update_energy())
    #result.append(test_update_resources())
    #result.append(test_update_position())
    result.append(compare_envs())
    # test if all results have passed
    if len(result) == sum(result):
        print("Success: All tests have passed")