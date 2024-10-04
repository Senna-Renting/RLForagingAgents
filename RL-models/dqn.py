# In parts borrowed this code from: https://github.com/erees1/jax-rl/tree/main

import jax.numpy as jnp
from jax import random, value_and_grad, jit, vmap
from jax.nn import one_hot
import model as m
from utils import ReplayBuffer

class DQN:
    def __init__(self, actions, lr=1e-4, eps=0.9, eps_min=0.1, eps_decay=0.99, gamma=0.98, buffer_size=10000, layer_spec=None, seed=0):
        self.lr = lr
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.key = random.PRNGKey(seed)
        self.batched_predict = lambda observations: m.batch_func(m.predict)(
            self.params, observations
        ) # Method that allows predictions for batches
        self.buffer = ReplayBuffer(self.buffer_size)
        self.steps_trained = 0
        self.actions = actions
        if layer_spec != None:
            self.params = m.init_network_params(layer_spec, self.key)
            self.layer_spec = layer_spec
    def act(self, observation, explore=True):
        self.key, subkey = random.split(self.key)
        # Update epsilon
        self.eps = max(self.eps*self.eps_decay,self.eps_min)
        if explore and random.uniform(self.key) < self.eps:
            action = random.randint(subkey, (), 0, self.layer_spec[-1])
        else:
            Q = m.predict(self.params, observation)
            action = jnp.argmax(Q)
        #print("Action taken: ", action)
        return action
        
    def update(self, batch_size):
        def get_Q_for_actions(params, observations):
            """Calculate Q values for action that was taken"""
            pred_Q_values = m.batch_func(m.predict)(params, observations)
            pred_Q_values = index_Q_at_action(pred_Q_values, actions)
            return pred_Q_values

        (
            obs,
            actions,
            r,
            next_obs,
            dones,
        ) = self.buffer.sample_batch(batch_size)

        max_next_Q_values = self.get_max_Q_values(next_obs)
        target_Q_values = self.get_target_Q_values(r, dones, max_next_Q_values)

        #  Caclulate loss and perform gradient descent
        loss, self.params = m.update(
            get_Q_for_actions, self.params, obs, target_Q_values, self.lr
        )
        self.steps_trained += 1 # Useful for fixed target version of DQN
        return loss

    def get_max_Q_values(self, next_obs):
        """Calculate max Q values for next state"""
        next_Q_values = self.batched_predict(next_obs)
        max_next_Q_values = jnp.max(next_Q_values, axis=-1)
        return max_next_Q_values

    def get_target_Q_values(self, rewards, dones, max_next_Q_values):
        """Calculate target Q values based on discounted max next_Q_values"""
        target_Q_values = (
            rewards + (1 - dones) * self.gamma * max_next_Q_values
        )
        return target_Q_values

# This implements an additional target network which is frozen over N iterations
class DQNFixedTarget(DQN):
    def __init__(self, actions, layer_spec=None, update_every=4, **kwargs):
        super().__init__(actions, layer_spec=layer_spec, **kwargs)
        self.update_every = update_every
        # Need to update key so target_params != params
        self.key = random.split(self.key)[0]
        if layer_spec is not None:
            self.target_params = m.init_network_params(layer_spec, self.key)

        # Target functions
        self.batched_predict_target = lambda observations: m.batch_func(m.predict)(
            self.target_params, observations
        )

    def get_max_Q_values(self, next_obs):
        if self.steps_trained % self.update_every == 0:
            # Jax arrays are immutable
            self.target_params = self.params
        next_Q_values = self.batched_predict_target(next_obs)
        max_next_Q_values = jnp.max(next_Q_values, axis=-1)
        return max_next_Q_values


# This makes sure that only taken actions in the batch samples are evaluated (it filters them out)
def index_Q_at_action(Q_values, actions):
    # Q_values [bsz, n_actions]
    # Actions [bsz,]
    idx = jnp.expand_dims(actions, -1).astype(int)
    # pred_Q_values [bsz,]
    pred_Q_values = jnp.take_along_axis(Q_values, idx, -1).squeeze()
    return pred_Q_values

