# Self-play using latest model inference. Gathering data for training. 

import os
import time
import json
import glob
import copy
import numpy as np

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker

from env import ClashRoyaleEnv, ReplayEnv
from takeAction import get_action_mask
from getState import createFinalState


NEXT_EPISODE = 60

TURN_TIME = 0.25
WAIT_AFTER_TURN = 0
TOTAL_EPISODES = 70

BUFFER_DIR = "buffers"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(BUFFER_DIR, exist_ok=True)

def save_episode(idx, left_transitions, right_transitions):
    """Save one episode to JSON."""
    path = f"{BUFFER_DIR}/ep_{idx}.json"

    def ser(t):
        return {
            "obs": t["obs"].tolist(),
            "action": t["action"],
            "reward": t["reward"],
            "next_obs": t["next_obs"].tolist(),
            "done": t["done"],
            "info": {
                "action_mask": t["info"]["action_mask"].tolist()
            }
        }

    data = {
        "left": [ser(t) for t in left_transitions],
        "right": [ser(t) for t in right_transitions],
    }

    with open(path, "w") as f:
        json.dump(data, f)


def load_checkpoint(env_left, env_right):
    files = sorted(glob.glob(f"{CHECKPOINT_DIR}/left_*.zip"), key=os.path.getmtime)

    if not files:
        print("No checkpoints found — starting fresh.")
        return None, None, 0

    latest = files[-1]
    ep = int(latest.split("_")[1].split(".")[0])

    print(f"Loading checkpoint from train_iter {ep}")

    left = MaskablePPO.load(latest)
    left.set_env(env_left)

    right = MaskablePPO.load(latest.replace("left", "right"))
    right.set_env(env_right)

    return left, right, ep

def reset_battle(env_left, env_right):
    env_left.reset()
    env_right.reset()

    env_left._start_battle()
    time.sleep(0.5)
    env_right._accept_battle()
    time.sleep(3)

    obs_left = env_left._get_obs()
    info_left = {"action_mask": get_action_mask(env_left.raw_state[2], env_left.raw_state[1])}

    obs_right = env_right._get_obs()
    info_right = {"action_mask": get_action_mask(env_right.raw_state[2], env_right.raw_state[1])}

    return obs_left, info_left, obs_right, info_right

# Training
env_left = ClashRoyaleEnv("left")
env_right = ClashRoyaleEnv("right")

model_left, model_right, last_train_ep = load_checkpoint(env_left, env_right)
start_ep = NEXT_EPISODE

if model_left is None:
    model_left = MaskablePPO(MaskableActorCriticPolicy, env_left, verbose=1)
    model_right = MaskablePPO(MaskableActorCriticPolicy, env_right, verbose=1)

print("Resetting initial battle...")
obs_left, info_left, obs_right, info_right = reset_battle(env_left, env_right)

print("Starting training...")
episode = start_ep
while episode < TOTAL_EPISODES:

    reward_left_sum = 0.0
    reward_right_sum = 0.0
    left_buf = []
    right_buf = []

    # One match
    while True:
        # LEFT TURN
        mask_left = info_left["action_mask"]
        act_left, _ = model_left.predict(obs_left, action_masks=mask_left, deterministic=False)

        t0 = time.time()
        next_left, r_left, done_left, _, next_info_left = env_left.step(act_left)
        reward_left_sum += r_left

        left_buf.append({
            "obs": obs_left,
            "action": int(act_left),
            "reward": float(r_left),
            "next_obs": next_left,
            "done": bool(done_left),
            "info": {"action_mask": mask_left},
        })

        obs_left, info_left = next_left, next_info_left

        if (elapsed := time.time() - t0) < TURN_TIME:
            time.sleep(TURN_TIME - elapsed)
        time.sleep(WAIT_AFTER_TURN)

        if done_left:
            print(f"Episode {episode} finished  (winner = {info_left.get('winner')})")
            createFinalState(info_left, info_right, obs_left, obs_right, left_buf, right_buf, True, False)
            break

        # RIGHT TURN
        mask_right = info_right["action_mask"]
        act_right, _ = model_right.predict(obs_right, action_masks=mask_right, deterministic=False)

        t0 = time.time()
        next_right, r_right, done_right, _, next_info_right = env_right.step(act_right)
        reward_right_sum += r_right

        right_buf.append({
            "obs": obs_right,
            "action": int(act_right),
            "reward": float(r_right),
            "next_obs": next_right,
            "done": bool(done_right),
            "info": {"action_mask": mask_right},
        })

        obs_right, info_right = next_right, next_info_right

        if (elapsed := time.time() - t0) < TURN_TIME:
            time.sleep(TURN_TIME - elapsed)
        time.sleep(WAIT_AFTER_TURN)

        if done_right:
            print(f"Episode {episode} finished  (winner = {info_right.get('winner')})")
            createFinalState(info_left, info_right, obs_left, obs_right, left_buf, right_buf, False, True)
            break

    print(f"Game {episode+1}: reward_left={reward_left_sum:.2f}, reward_right={reward_right_sum:.2f}")
    save_episode(episode, left_buf, right_buf)
    episode += 1

    time.sleep(2)
    obs_left, info_left, obs_right, info_right = reset_battle(env_left, env_right)
    time.sleep(1)

print("Data collection finished")

