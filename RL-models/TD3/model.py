import jax.numpy as jnp
from jax import random
import jax
from flax import nnx
import optax
from buffer import ReplayBuffer
from parameters import generate_value_network, generate_policy_network
from functools import partial

class TD3:
    def __init__(self, seed, state_dim, action_dim, 
                 buffer_size, action_max, policy_interval=2,
                 tau=0.01, sigma=0.2, gamma=0.99, lr_Q=1e-3, lr_P=3e-4, noise_clip=0.5, sigma_explore=0.2, update_interval=1):
        self.buffer = ReplayBuffer(state_dim, action_dim, buffer_size)
        self.action_max = action_max
        self.policy_interval = policy_interval
        self.tau = tau
        self.sigma = sigma
        self.sigma_explore = sigma_explore
        self.gamma = gamma
        self.update_interval = update_interval
        self.params_rng = nnx.Rngs(seed)
        self.total_it = 0
        self.noise_clip = noise_clip
        self.Qnet1 = generate_value_network(self.params_rng, state_dim, action_dim)
        self.Qnet2 = generate_value_network(self.params_rng, state_dim, action_dim)
        self.Pnet = generate_policy_network(self.params_rng, state_dim, action_dim, action_max)
        self.Qnet1_target = generate_value_network(self.params_rng, state_dim, action_dim)
        self.Qnet2_target = generate_value_network(self.params_rng, state_dim, action_dim)
        self.Pnet_target = generate_policy_network(self.params_rng, state_dim, action_dim, action_max)
        # Make sure targets are equal in state w.r.t their no-target versions
        _, state = nnx.split(self.Qnet1)
        nnx.update(self.Qnet1_target,state)
        _, state = nnx.split(self.Qnet2)
        nnx.update(self.Qnet2_target,state)
        _, state = nnx.split(self.Pnet)
        nnx.update(self.Pnet_target,state)
        # Set unique optimizers for each network (Note: adam uses stochastic gradient DESCENT)
        self.optim_Qnet1 = nnx.Optimizer(self.Qnet1, optax.adam(lr_Q))
        self.optim_Qnet2 = nnx.Optimizer(self.Qnet2, optax.adam(lr_Q))
        self.optim_Pnet = nnx.Optimizer(self.Pnet, optax.adam(lr_P))

    def select_action(self, state):
        return self.Pnet_target(state)

    #@partial(jax.jit, static_argnums=(0))
    def sample_action(self, rng, state):
        mu = self.select_action(state)
        rng, subkey = random.split(rng)
        action = mu + self.sigma_explore*self.action_max*random.normal(subkey, mu.shape)
        return jnp.clip(action, -self.action_max, self.action_max)

    #@partial(jax.jit, static_argnums=(0))
    def select_target_action(self, rng, state):
        mu = self.select_action(state)
        rng, subkey = random.split(rng)
        action = mu + jnp.clip(self.sigma*self.action_max*random.normal(subkey, mu.shape), -self.noise_clip, self.noise_clip)
        return jnp.clip(action, -self.action_max, self.action_max)

    
    def train(self, rng, batch_size):
        self.total_it += 1
        rng, buffer_key, crit_key = random.split(rng, 3)
        buffer_out = self.buffer.sample(buffer_key, batch_size)
        loss_Q1, loss_Q2 = self._update_critics(crit_key, buffer_out)
        if self.total_it % self.policy_interval == 0:
            state, action, *_ = buffer_out
            loss_actor = self._update_actor(state)
            nnx.update(self.Qnet1_target, self._polyak_update(self.Qnet1_target, self.Qnet1))
            nnx.update(self.Qnet2_target, self._polyak_update(self.Qnet2_target, self.Qnet2))
            nnx.update(self.Pnet_target, self._polyak_update(self.Pnet_target, self.Pnet))
            return loss_Q1, loss_Q2, loss_actor
        return loss_Q1, loss_Q2, None

    #@partial(jax.jit, static_argnums=(0))
    def _update_critics(self, rng, buffer_out):
        _, action, next_state, reward, done = buffer_out
        rng, subkey = random.split(rng, 2)
        next_action = self.select_target_action(subkey, next_state)
        ys = reward + (1-done)*self.gamma*jnp.min(
                                          jnp.array([self.Qnet1_target(next_state,next_action),                                                               self.Qnet2_target(next_state,next_action)]).flatten())
        loss_fn = lambda model: ((model(next_state, next_action) - ys) ** 2).mean()
        loss_Q1, grads_Q1 = nnx.value_and_grad(loss_fn)(self.Qnet1)
        loss_Q2, grads_Q2 = nnx.value_and_grad(loss_fn)(self.Qnet2)
        self.optim_Qnet1.update(grads_Q1)
        self.optim_Qnet2.update(grads_Q2)
        return loss_Q1, loss_Q2

    #@partial(jax.jit, static_argnums=(0,))
    def _update_actor(self, state):
        # -1 product guarantees gradient ascent
        loss_fn = lambda Pmodel, Qmodel: (-1*Qmodel(state, Pmodel(state))).mean()
        # Differentiate w.r.t params of Pnet
        loss, grads = nnx.value_and_grad(loss_fn)(self.Pnet, self.Qnet1)
        # print(jax.tree.map(lambda x: x.flatten().mean(), grads)) <- Proofs gradients are not zero
        self.optim_Pnet.update(grads)
        return loss

    #@partial(jax.jit, static_argnums=(0,))
    def _polyak_update(self, model_target, model_current):
        target_params = nnx.state(model_target)
        current_params = nnx.state(model_current)
        update_fn = lambda p_current, p_target: self.tau * p_current + (1-self.tau) * p_target
        return jax.tree.map(update_fn, current_params, target_params)

    #@partial(jax.jit, static_argnums=(0,))
    def get_policy(self):
        return nnx.state(self.Pnet_target)
        

if __name__ == "__main__":
    rngs = nnx.Rngs(0)
    model = TD3(0,3,1,1000,3)
    state = jnp.array([1,2,3])
    action = jnp.array([1])
    print(model.Qnet2.l1)
        