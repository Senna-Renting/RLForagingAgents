import orbax.checkpoint as ocp
from flax import nnx
import os
import jax
orbax_checkpointer = ocp.StandardCheckpointer()

def load_policy(model, wandb_code):
    state = nnx.state(model)
    restored_model = orbax_checkpointer.restore(wandb_code, item=state)
    nnx.update(model, restored_model)

def save_policy(model, wandb_code):
    state = nnx.state(model)
    orbax_checkpointer.save(wandb_code, state)

# Below two functions will be used for our re-train implementation
def save_policies(models, type, path):
    for i, model in enumerate(models):
        graphdef, state = nnx.split(model)
        orbax_checkpointer.save(os.path.abspath(os.path.join(path, type, f"A{i}")), state)

def load_policies(models, type, path):
    for i, model in enumerate(models):
        graphdef, state = nnx.split(model)
        restored_model = orbax_checkpointer.restore(os.path.join(path, type, f"A{i}"), state)
        nnx.update(model, restored_model)
    