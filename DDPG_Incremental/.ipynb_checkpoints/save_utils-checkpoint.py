import orbax.checkpoint
from flax import nnx
orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

def load_policy(model, wandb_code):
    state = nnx.state(model)
    restored_model = orbax_checkpointer.restore(wandb_code, item=state)
    nnx.update(model, restored_model)

def save_policy(model, wandb_code):
    state = nnx.state(model)
    orbax_checkpointer.save(wandb_code, state)
    