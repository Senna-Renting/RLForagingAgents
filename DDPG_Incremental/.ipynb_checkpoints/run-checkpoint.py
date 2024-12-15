import wandb
from environment import *
from save_utils import load_policy, save_policy
from td3 import Actor, wandb_train_ddpg, train_ddpg
import os

# This python file should contain all the high-level training function variations of our environment
def ddpg_train_patch(env, num_episodes):
    episodes = list(range(1,num_episodes+1))
    action_dim, a_range = env.get_action_space()
    lr_c = 0.0004185607534989185
    lr_a = 0.00003732562723230332
    tau = 0.03630053588901042
    rewards, actor, critic, reset_key = train_ddpg(env, episodes[-1], lr_c=lr_c, lr_a=lr_a, tau=tau, action_dim=action_dim, state_dim=env.get_state_space()[1], action_max=a_range[1], hidden_dim=256, batch_size=200, seed=0, reset_seed=0)
    input("Press enter to see trained model in action...")
    env = RenderOneAgentEnvironment(env)
    state, info = env.reset(seed=reset_key)
    while True:
        state, reward, terminated, truncated, _ = env.step(actor(state))
        if terminated or truncated:
            break
    env.render()

def wandb_ddpg_train_patch(env, num_episodes, num_runs=5, hidden_dim=32, batch_size=100, warmup_steps=200):
    episodes = list(range(1,num_episodes+1))
    action_dim, a_range = env.get_action_space()
    wandb.login()
    sweep_config = {
        'method':'bayes',
        'metric':{
            'name':'Return',
            'goal':'maximize'
        },
        'parameters':{
            'lr_c':{
                'distribution':'uniform',
                'min': 5e-4,
                'max': 3e-3
            },
            'lr_a':{
                'distribution':'uniform',
                'min': 5e-5,
                'max': 5e-4
            },
            'tau':{
                'distribution':'uniform',
                'min':0,
                'max':0.3
            },
            'action_dim':{
                'value':action_dim
            },
            'state_dim':{
                'value':env.get_state_space()[1]
            },
            'hidden_dim':{
                'value':hidden_dim
            },
            'batch_size':{
                'value':batch_size
            },
            'num_episodes':{
                'value':num_episodes
            },
            'warmup_steps':{
                'value':warmup_steps
            },
            'seed':{
                'value':0
            },
            'reset_seed':{
                'value':0
            },
            'action_max':{
                'value':a_range[1]
            }
        }
    }
    sweep_id = wandb.sweep(sweep_config, project="OneAgentPatchDDPG")
    train_fun = wandb_train_ddpg(env)
    wandb.agent(sweep_id, train_fun, count=num_runs)

def patch_test_saved_policy(env, path, hidden_dim=32):
    state_dim = env.get_state_space()[1]
    action_dim, action_max = env.get_action_space()
    policy = Actor(state_dim, action_dim, action_max[1], 0, hidden_dim=hidden_dim)
    load_policy(policy, path)
    env = RenderOneAgentEnvironment(env)
    state, info = env.reset(seed=0)
    while True:
        state, reward, terminated, truncated, _ = env.step(policy(state))
        if terminated or truncated:
            break
    env.render()

if __name__ == "__main__":
    env = OneAgentEnv(patch_radius=0.5, step_max=400, alpha=2)
    num_episodes = 100
    num_runs = 5
    # Uncomment the method needed below
    ddpg_train_patch(env, num_episodes)
    #wandb_ddpg_train_patch(env, num_episodes, num_runs=num_runs, hidden_dim=256, batch_size=100, warmup_steps=200)
    # Fill in the path of the policy and uncomment the method below it
    #path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policies", "e4nii8kg", "efficient-sweep-4")
    #patch_test_saved_policy(env, path, hidden_dim=256)