from model import TD3
import gymnasium as gym

SEED = 42

def gym_train_td3(env, num_episodes, batch_size, buffer_size, warmup_steps, **kwargs):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_max = env.action_space.high[0] # Might need to refine this to generalize better
    td3 = TD3(SEED, state_dim, action_dim, buffer_size, action_max, **kwargs)
    returns = []
    loss_Q1 = []
    loss_Q2 = []
    loss_actor = []
    state, info = env.reset(seed=SEED)
    # Warm-up period
    for i in range(warmup_steps):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        done_w = terminated or truncated
        if done_w: 
            env.reset(seed=SEED)
        td3.buffer.add(state, action, next_state, reward, int(done_w))
    # Training loop
    for i in range(1,num_episodes+1):
        avg_return = 0
        avg_loss_Q1 = 0
        avg_loss_Q2 = 0
        avg_loss_actor = 0
        n_samples = 0
        state, info = env.reset(seed=SEED)
        done = False
        while done != True:
            # Perform an action and add to buffer
            action = td3.select_action(state).clip(-action_max, action_max)
            next_state, reward, terminated, truncated, info = env.step(action)
            avg_return += reward
            n_samples += 1
            #print("Terminated? ", terminated)
            #print("Truncated? ", truncated)
            done = terminated or truncated
            #print("Done? ", done, end="\n\n")
            td3.buffer.add(state, action, next_state, reward, int(terminated))
            ep_loss_Q1, ep_loss_Q2, ep_loss_actor = td3.train(batch_size)
            avg_loss_Q1 += ep_loss_Q1
            avg_loss_Q2 += ep_loss_Q2
            if ep_loss_actor is not None:
                avg_loss_actor += ep_loss_actor
        avg_return /= n_samples
        avg_loss_Q1 /= n_samples
        avg_loss_Q2 /= n_samples
        avg_loss_actor /= n_samples
        returns.append(avg_return)
        loss_Q1.append(avg_loss_Q1)
        loss_Q2.append(avg_loss_Q2)
        loss_actor.append(avg_loss_actor)
        print("Episode: ", i)
        print("Average return: ", avg_return)
        print("Average loss Q1: ", avg_loss_Q1)
        print("Average loss Q2: ", avg_loss_Q2)
        print("Average loss actor: ", avg_loss_actor)
    return td3, returns, loss_Q1, loss_Q2, loss_actor

def gym_test_td3(env, model):
    state, info = env.reset(seed=SEED)
    done = False
    while not done:
        state, reward, terminated, truncated, info = env.step(model.select_action(state))
        done = terminated or truncated
        

if __name__ == "__main__":
    env = gym.make("Pendulum-v1")
    gym_train_td3(env, 10, 20, 2000, 100)