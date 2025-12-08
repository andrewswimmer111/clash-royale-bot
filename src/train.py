# train_ppo.py — offline PPO training from buffers

import os
import json
import glob
import copy
import numpy as np
import torch

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker

from env import ReplayEnv   # cleaned ReplayEnv with correct spaces


BUFFER_DIR = "buffers"
CHECKPOINT_DIR = "checkpoints"
MAX_BUFFER_EPISODES = 50
MIN_LEARN_STEPS = 64
CHECKPOINT_INTERVAL = 10

MAX_TRAIN_ITERS = 500
PAT = 5          # patience
THR = 1e-3       # loss threshold
MIN_STEPS = 50   # warm-up

best_left_loss = float('inf')
best_right_loss = float('inf')
left_patience_counter = 0
right_patience_counter = 0

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Load episode JSON
def load_episode_json(path):
    with open(path, "r") as f:
        raw = json.load(f)

    def deser(t):
        return {
            "obs": np.array(t["obs"], dtype=np.float32),
            "action": t["action"],
            "reward": float(t["reward"]),
            "next_obs": np.array(t["next_obs"], dtype=np.float32),
            "done": bool(t["done"]),
            "info": {
                "action_mask": np.array(t["info"]["action_mask"], dtype=bool)
            },
        }

    left = [deser(t) for t in raw["left"]]
    right = [deser(t) for t in raw["right"]]
    return left, right


def load_recent_episodes():
    eps = sorted(glob.glob(f"{BUFFER_DIR}/ep_*.json"))
    eps = eps[-MAX_BUFFER_EPISODES:]
    buffers = []

    for file in eps:
        left, right = load_episode_json(file)
        buffers.append({"left": left, "right": right})
    return buffers


def load_latest_checkpoint():
    files = sorted(glob.glob(f"{CHECKPOINT_DIR}/left_*.zip"))
    if not files:
        print("No checkpoints found — training from scratch.")
        return None, None, 0

    latest_left = files[-1]
    latest_right = latest_left.replace("left_", "right_")

    filename = os.path.basename(latest_left)
    ep_num = int(filename.split("_")[1].split(".")[0])
    print(f"Loading checkpoint: {latest_left} (train_iter {ep_num})")

    left_model = MaskablePPO.load(latest_left)
    right_model = MaskablePPO.load(latest_right)

    return left_model, right_model, ep_num

# Mask function for ActionMasker
def make_mask_fn(replay_env):
    def mask_fn(obs):
        idx = replay_env.pos
        if idx >= len(replay_env.transitions):
            idx = len(replay_env.transitions) - 1
        return replay_env.transitions[idx]["info"]["action_mask"]
    return mask_fn

# Initialize Models
model_left, model_right, START_CHECKPOINT = load_latest_checkpoint()

if model_left is None:
    dummy_env = ReplayEnv([{
        "obs": np.zeros(524, dtype=np.float32),
        "action": 0,
        "reward": 0.0,
        "next_obs": np.zeros(524, dtype=np.float32),
        "done": True,
        "info": {"action_mask": np.ones(97, dtype=bool)},
    }])
    model_left = MaskablePPO(MaskableActorCriticPolicy, dummy_env, verbose=0)
    model_right = MaskablePPO(MaskableActorCriticPolicy, dummy_env, verbose=0)

# Load buffers and setup env
buffers = load_recent_episodes()
    
print(f"Loaded {len(buffers)} episodes for training.")

all_left = []
for ep in buffers:
    all_left.extend(ep["left"])

replay_left = ReplayEnv(all_left)
mask_fn_left = make_mask_fn(replay_left)
wrapped_left = ActionMasker(replay_left, mask_fn_left)

model_left.set_env(wrapped_left)
learn_steps_l = max(len(all_left), MIN_LEARN_STEPS)

all_right = []
for ep in buffers:
    all_right.extend(ep["right"])

replay_right = ReplayEnv(all_right)
mask_fn_right = make_mask_fn(replay_right)
wrapped_right = ActionMasker(replay_right, mask_fn_right)

model_right.set_env(wrapped_right)
learn_steps_r = max(len(all_right), MIN_LEARN_STEPS)

print(f"{'Step':<8} {'Left Loss':<12} {'Right Loss':<12}")
print("-" * 35)

# Training Loop
for train_step in range(START_CHECKPOINT, MAX_TRAIN_ITERS):

    if not buffers:
        print("No episodes available.")
        break

    # Left
    model_left.learn(total_timesteps=learn_steps_l, reset_num_timesteps=False)

    # Patience
    left_loss = model_left.logger.name_to_value.get("train/value_loss", None)

    if left_loss < best_left_loss - THR:
        best_left_loss = left_loss
        left_patience_counter = 0
    else:
        left_patience_counter += 1

    # Right
    model_right.learn(total_timesteps=learn_steps_r, reset_num_timesteps=False)

    # Patience
    right_loss = model_right.logger.name_to_value.get("train/value_loss", None)

    if right_loss < best_right_loss - THR:
        best_right_loss = right_loss
        right_patience_counter = 0
    else:
        right_patience_counter += 1

    # Logs:
    print(f"{train_step:<8} {left_loss:<12.6f} {right_loss:<12.6f}")

    # Save checkpoints
    if train_step != START_CHECKPOINT and train_step % CHECKPOINT_INTERVAL == 0:
        model_left.save(f"{CHECKPOINT_DIR}/left_{train_step}.zip")
        model_right.save(f"{CHECKPOINT_DIR}/right_{train_step}.zip")
        print(f"[Checkpoint] Saved at train step {train_step}")

    # Early stopping
    if (train_step > MIN_STEPS and left_patience_counter >= PAT and right_patience_counter >= PAT):
        print(f"Training converged at step {train_step}")
        break

print("Training finished.")
