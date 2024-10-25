from model import TD3
from save_utils import save_policy, load_policy
import gymnasium as gym
from gym.wrappers import TimeLimit
import gymnax
import jax
import jax.numpy as jnp
import wandb
import numpy as np
from flax import nnx
from parameters import *

SEED = 42
SEEDS = list(range(50))
np.random.seed(SEED)

def train_td3(env, num_episodes, batch_size, buffer_size, warmup_steps, **kwargs):
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

def test_td3(env, model):
    state, info = env.reset(seed=SEED)
    done = False
    while not done:
        state, reward, terminated, truncated, info = env.step(model.select_action(state))
        done = terminated or truncated

def wandb_train_td3(config=None):
    rng = jax.random.PRNGKey(SEED)
    with wandb.init(config=config) as run:
        config = wandb.config
        env, env_params = gymnax.make("Pendulum-v1")
        state_dim = env.observation_space(env_params).shape[0]
        action_dim = env.action_space(env_params).shape[0]
        action_max = env.action_space(env_params).high # Might need to refine this to generalize better
        td3 = TD3(SEED, state_dim, action_dim, config.buffer_size, action_max, gamma=config.gamma, sigma=config.sigma, sigma_explore=config.sigma_explore, policy_interval=config.policy_interval, tau=config.tau, noise_clip=config.noise_clip, lr_Q=config.lr_critic, lr_P=config.lr_actor)
        # Keep track of the best return
        best_avg_return = -9999
        best_policy = td3.get_policy()

        rng, key_reset, key_actions = jax.random.split(rng, 3)
        actions = jax.random.uniform(key_actions, shape=config.warmup_amount, minval=-action_max, maxval=action_max)
        obs, state = env.reset(key_reset, env_params)
        # Sample randomly for the buffer before training
        for i in range(config.warmup_amount):
            rng, key_step, key_reset = jax.random.split(rng, 3)
            next_obs, next_state, reward, done, _ = env.step(key_step, state, actions[i], env_params)
            td3.buffer.add(obs, actions[i], next_obs, reward, 0)
            obs = next_obs
            # Jax formulation for: when done reset the environment
            obs, state = jax.lax.cond(done, env.reset, lambda key, env_params: (obs, state), key_reset, env_params)
        
        # Training loop
        for epoch in range(1,config.num_episodes+1):
            avg_loss_Q1 = 0
            avg_loss_Q2 = 0
            avg_loss_actor = 0
            n_samples = 0
            rng, key_reset = jax.random.split(rng, 2)
            obs, state = env.reset(key_reset, env_params)
            done = False
            while done != True:
                # Perform an action and add to buffer
                rng, key_step, key_action, key_train = jax.random.split(rng, 4)
                action = td3.sample_action(key_action, obs)
                next_obs, next_state, reward, done, _ = env.step(key_step, state, action, env_params)
                n_samples += 1
                td3.buffer.add(obs, action, next_obs, reward, 0)
                obs = next_obs
                state = next_state
                ep_loss_Q1, ep_loss_Q2, ep_loss_actor = td3.train(key_train, config.batch_size)
                avg_loss_Q1 += ep_loss_Q1.item()
                avg_loss_Q2 += ep_loss_Q2.item()
                # Jax formulation of: only add if present
                ep_loss_actor = ep_loss_actor.item() if ep_loss_actor is not None else 0
                avg_loss_actor += ep_loss_actor
                #avg_loss_actor = jax.lax.cond(ep_loss_actor is not None, lambda avg, ep: avg+ep.item(), lambda avg, ep: avg, avg_loss_actor, ep_loss_actor)
            # Normalize items to their averages
            avg_loss_Q1 /= n_samples
            avg_loss_Q2 /= n_samples
            avg_loss_actor /= (n_samples / config.policy_interval)
            avg_return, max_reward = wandb_test_td3(td3, max_length_episodes=config.max_length_episode)
            # Model saving logic
            #new_policy = lambda best_avg, avg, best_policy, policy: (avg, policy)
            #same_policy = lambda best_avg, avg, best_policy, policy: (best_avg, best_policy)
            #best_avg_return, best_policy = jax.lax.cond(avg_return > best_avg_return, new_policy, same_policy, best_avg_return, avg_return, best_policy, td3.get_policy())
            if avg_return > best_avg_return:
                best_policy = td3.get_policy()
                best_avg_return = avg_return
            # Log the metrics to wandb project file
            wandb.log({"Epoch":epoch, 
               "Qnet1 average loss": avg_loss_Q1, 
               "Qnet2 average loss": avg_loss_Q2, 
               "Actor average loss": avg_loss_actor,
               "Average return": avg_return,
               "Highest reward": max_reward})
        save_policy(best_policy, run.sweep_id+"/"+run.name) # Save the best policy to a unique folder

def wandb_test_td3(model, max_length_episodes=200):
    rng = jax.random.PRNGKey(SEED)
    env, env_params = gymnax.make("Pendulum-v1")
    rng, key_reset = jax.random.split(rng, 2)
    obs, state = env.reset(key_reset, env_params)
    done = False
    avg_reward = 0
    max_reward = -9999
    n_samples = 0
    for i in range(max_length_episodes):
        rng, key_action = jax.random.split(rng, 2)
        n_samples += 1
        next_obs, next_state, reward, done, _ = env.step(key_action, state, model.select_action(obs), env_params)
        state = next_state
        obs = next_obs
        avg_reward += reward
        if reward > max_reward:
            max_reward = reward
        if done:
            break
    return avg_reward / n_samples, max_reward # Return the average reward over one episode

def wandb_test_saved_td3(wandb_code):
    env = gym.make("Pendulum-v1", render_mode="human")
    state, info = env.reset(seed=SEED)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_max = env.action_space.high[0] 
    policy = generate_policy_network(nnx.Rngs(SEED), state_dim, action_dim, action_max)
    load_policy(policy, wandb_code)
    done = False
    i = 0
    for i in range(500):
        next_state, reward, terminated, truncated, info = env.step(policy(state))
        state = next_state
        if terminated:
            break
    env.close()

def random_run(num_episodes, max_length_episodes=200, seed=SEED):
    np.random.seed(seed)
    env = TimeLimit(gym.make("Pendulum-v1"), max_episode_steps=max_length_episodes)
    avg_reward = 0
    n_samples = 0
    actions = np.random.uniform(-2,2,size=max_length_episodes*num_episodes)
    for i in range(num_episodes):
        state, info = env.reset(seed=SEED)
        done = False
        while not done:
            next_state, reward, terminated, truncated, info = env.step([actions[n_samples]])
            state = next_state
            avg_reward += reward
            n_samples += 1
            done = terminated or truncated
    return avg_reward / n_samples # Return the average reward over one episode
        

if __name__ == "__main__":
    env = gym.make("Pendulum-v1")
    gym_train_td3(env, 10, 20, 2000, 100)