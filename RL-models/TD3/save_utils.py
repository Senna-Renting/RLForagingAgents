from parameters import generate_policy_network
import orbax.checkpoint
from flax import nnx

PATH = "/td3-pendulum/"
orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

def load_policy(model, wandb_code):
    state = nnx.state(model)
    restored_model = orbax_checkpointer.restore(PATH+wandb_code, item=state)
    nnx.update(model, restored_model)

def save_policy(model, wandb_code):
    orbax_checkpointer.save(PATH+wandb_code, model)