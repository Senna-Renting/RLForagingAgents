from environment import *

def test_env_initialization():
    env_state = initialize_env(2, True, zero_sw, 0)
    env_state = initialize_patch(env_state)
    env_state = initialize_agents(jax.random.PRNGKey(0), env_state)
    print(env_state.get("agents_state").get("position"), env_state.get("patch_state").get("position"))
    print(get_env_obs(env_state))
    return True

if __name__ == "__main__":
    result = []
    # add all the tests below
    result.append(test_env_initialization())
    # test if all results have passed
    if len(result) == sum(result):
        print("Success: All tests have passed")