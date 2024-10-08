import gymnasium as gym
from dqn import DQN
import jax.numpy as jnp
import matplotlib.pyplot
from IPython.display import clear_output # For clearing logging

# I removed the logging from this function, might add it back later to check if I can make the whole thing work
def run(
    env,
    agent,
    training=True,
    ep_steps=20,
    render=False,
    warm_up_eps=0,
    seed=0,
    **kwargs,
):
    ep_rewards = []
    ep_losses = []
    total_steps = 0

    for i_episode in range(int(ep_steps + warm_up_eps)):
        observation, info = env.reset()
        ep_reward = 0
        ep_loss = []
        done = False
        t = 0

        while not done:
            if render:
                env.render()

            # Step environment and add to buffer
            observation, reward, done, info = play_one_step(
                env, agent, observation, training
            )

            # Update model if training
            if training and i_episode > warm_up_eps:
                loss = agent.update(kwargs["batch_size"])
                ep_loss.append(loss)

            # Update counters:
            ep_reward += reward
            t += 1
            total_steps += 1

        ep_rewards.append(ep_reward)

        clear_output(wait=True)
        print("Episode: ", i_episode)
        print("Reward: ", ep_reward)
        if training and i_episode > warm_up_eps:
            ep_mean_loss = jnp.array(ep_loss).mean()
            ep_losses.append(ep_mean_loss)
            print("Mean loss: ", ep_mean_loss)
        
        
    env.close()
    return ep_rewards, ep_losses, agent

def play_one_step(env, agent, observation, training=False):
    action = agent.actions[agent.act(observation, training)]
    next_observation, reward, terminated, truncated, info = env.step(jnp.array([action]))
    done = terminated or truncated
    if training:
        agent.buffer.append((observation, action, reward, next_observation, done))

    return next_observation, reward, done, info

def train(env, agent, train_eps=200, **kwargs):
    rewards, losses, agent = run(env, agent, ep_steps=train_eps, **kwargs)
    return rewards, losses, agent

def test(env, agent, test_eps=100, warm_up_eps=0, **kwargs):
    return run(
        env, agent, render=True, training=False, warm_up_eps=0, ep_steps=test_eps, **kwargs
    )[0]
    