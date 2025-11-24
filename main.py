from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

from enviroment.calico_env import CalicoEnv
from models.one_step_lookahead import one_step_lookahead_move

app = FastAPI()
class Action(BaseModel):
    action_type: str
    tile_idx: int
    row: int | None = None
    col: int | None = None


class StepRequest(BaseModel):
    flat_state: list
    action: Action | None = None


class StepResponse(BaseModel):
    new_state: list
    used_action: tuple


# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------

def build_env_from_state(flat_state):
    env = CalicoEnv()
    env.start_game()
    env.set_from_flat_state(np.array(flat_state))
    return env


# ------------------------------------------------------------
# FASTAPI ENDPOINT
# ------------------------------------------------------------

@app.post("/one-step-lookahead-move", response_model=StepResponse)
def step(req: StepRequest):
    print(req.flat_state)
    print(req)
    print(type(req.flat_state))
    print(len(req.flat_state))
    # rebuild environment

    env = build_env_from_state(req.flat_state)
    env.set_selected_from_empty()
    print(env)
    used_action = one_step_lookahead_move(env)
    print(env)
    if env.mode == "placing":
        used_action = (used_action,0,0)

    # get updated state
    new_state = env.get_flat_state().tolist()
    print(new_state)
    print(used_action)
    return StepResponse(
        new_state=new_state,
        used_action=used_action
    )

