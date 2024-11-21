from environment import *

def test_state_shape():
    env = Environment()
    check = env.get_state_space() == (1,4)
    if not check:
        print("Error: State space not in the correct shape")
    return check

def test_reset():
    env = Environment()
    state, _ = env.reset(seed=10)
    check1 = state.shape == (1,4)
    check2 = env._dist_to_patch() > env.patch_radius
    check3 = env.step_idx == 0
    check = check1 and check2 and check3
    if not check:
        print("Error: Reset function not behaving correctly")
    return check

def test_step_pos_change():
    env = Environment(agent_step=1, x_max=10, y_max=10)
    env.step(jnp.array([0])) # Step right >
    check1 = jnp.all(env.agent_pos == jnp.array([6,5]))
    env.step(jnp.array([0.5*jnp.pi])) # Step down v
    check2 = jnp.all(env.agent_pos == jnp.array([6,6]))
    check = check1 and check2
    if not check:
        print("Error: Step function not behaving correctly")
    return check

def test_step_patch():
    # Test stationary point
    env = Environment(agent_step=0, s_init=10, eta=11, gamma=1.1)
    env.reset()
    env.step(jnp.array([0]))
    check1 = env.s_patch - 10 < 1e-5 # Correction for floating point instability
    # Test collapse of resource amount
    env = Environment(agent_step=0, s_init=10, eta=11, gamma=1.2, )
    env.reset()
    env.step(jnp.array([0]))
    check2 = env.s_patch - 10 < 0
    # Test explosion of resource amount
    env = Environment(agent_step=0, s_init=10, eta=11, gamma=1, beta=0)
    env.reset()
    env.step(jnp.array([0]))
    check3 = env.s_patch - 10 > 0
    # Test depletion of stationary patch by agent
    env = Environment(agent_step=0, s_init=10, eta=11, gamma=1.1, beta=1)
    env.step(jnp.array([0]))
    check4 = env.s_patch - 10 < 0
    # Test stable point with eating included
    env = Environment(agent_step=0, s_init=10, eta=12, gamma=1.1, beta=1, e_init=0, e_max=10)
    env.step(jnp.array([0]))
    check5 = env.s_patch - 10 < 1e-5
    check = check1 and check2 and check3 and check4 and check5
    if not check:
        print("Error: Patch resource not updated correctly")
    return check
    

if __name__ == "__main__":
    result = []
    # add all the tests below
    result.append(test_state_shape())
    result.append(test_reset())
    result.append(test_step_pos_change())
    result.append(test_step_patch())
    # test if all results have passed
    if len(result) == sum(result):
        print("Success: All tests have passed")