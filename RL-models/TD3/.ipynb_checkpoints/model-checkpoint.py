import jax.numpy as jnp
from jax import random
import jax
from flax import nnx
import optax
from buffer import ReplayBuffer
from parameters import generate_value_network, generate_policy_network

class TD3:
    def __init__(self, seed, state_dim, action_dim, 
                 buffer_size, action_max, policy_interval=50,
                 tau=0.01, sigma=0.01, gamma=0.99, lr=3e-5, noise_clip=0.5):
        print(lr)
        self.buffer = ReplayBuffer(state_dim, action_dim, buffer_size)
        self.action_max = action_max
        self.policy_interval = policy_interval
        self.tau = tau
        self.sigma = sigma
        self.gamma = gamma
        self.params_rng = nnx.Rngs(seed)
        self.rng = random.PRNGKey(seed)
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
        self.optim_Qnet1 = nnx.Optimizer(self.Qnet1, optax.adam(lr))
        self.optim_Qnet2 = nnx.Optimizer(self.Qnet2, optax.adam(lr))
        self.optim_Pnet = nnx.Optimizer(self.Pnet, optax.adam(lr))

    def select_action(self, state):
        return self.Pnet_target(state)

    def sample_action(self, state):
        mu = self.select_action(state)
        self.rng, subkey = random.split(self.rng)
        action = mu + self.action_max * self.sigma * random.normal(subkey, mu.shape)
        return jnp.clip(action, -self.action_max, self.action_max)

    def train(self, batch_size):
        self.total_it += 1
        self.rng, subkey = random.split(self.rng)
        buffer_out = self.buffer.sample(subkey, batch_size)
        loss_Q1, loss_Q2 = self._update_critics(buffer_out)
        if self.total_it % self.policy_interval == 0:
            state, action, *_ = buffer_out
            loss_actor = self._update_actor(state)
            nnx.update(self.Qnet1_target, self._polyak_update(self.Qnet1_target, self.Qnet1))
            nnx.update(self.Qnet2_target, self._polyak_update(self.Qnet2_target, self.Qnet2))
            nnx.update(self.Pnet_target, self._polyak_update(self.Pnet_target, self.Pnet))
            return loss_Q1, loss_Q2, loss_actor
        return loss_Q1, loss_Q2, None

    def _update_critics(self, buffer_out):
        _, action, next_state, reward, done = buffer_out
        self.rng, subkey = random.split(self.rng)
        noise = (random.normal(subkey, action.shape)*self.sigma).clip(-self.noise_clip, self.noise_clip)
        next_action = self.Pnet_target(next_state) + noise
        next_action = jnp.clip(next_action, -self.action_max, self.action_max)
        ys = reward + (1-done)*self.gamma*jnp.min(
                                          jnp.array([self.Qnet1_target(next_state,next_action),                                                               self.Qnet2_target(next_state,next_action)]).flatten())
        loss_fn = lambda model: ((model(next_state, next_action) - ys) ** 2).mean()
        loss_Q1, grads_Q1 = nnx.value_and_grad(loss_fn)(self.Qnet1)
        loss_Q2, grads_Q2 = nnx.value_and_grad(loss_fn)(self.Qnet2)
        self.optim_Qnet1.update(grads_Q1)
        self.optim_Qnet2.update(grads_Q2)
        return loss_Q1.item(), loss_Q2.item()

    def _update_actor(self, state):
        # -1 product guarantees gradient ascent
        loss_fn = lambda Pmodel, Qmodel: (-1*Qmodel(state, Pmodel(state))).mean()
        # Differentiate w.r.t params of Pnet
        loss, grads = nnx.value_and_grad(loss_fn)(self.Pnet, self.Qnet1)
        # print(jax.tree.map(lambda x: x.flatten().mean(), grads)) <- Proofs gradients are not zero
        self.optim_Pnet.update(grads)
        return loss.item()

    def _polyak_update(self, model_target, model_current):
        target_params = nnx.state(model_target)
        current_params = nnx.state(model_current)
        update_fn = lambda p_current, p_target: self.tau * p_current + (1-self.tau) * p_target
        return jax.tree.map(update_fn, current_params, target_params)
        

if __name__ == "__main__":
    rngs = nnx.Rngs(0)
    model = TD3(0,3,1,1000,3)
    state = jnp.array([1,2,3])
    action = jnp.array([1])
    print(model.Qnet2.l1)
        