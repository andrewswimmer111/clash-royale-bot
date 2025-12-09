# Evaluate trained checkpoint against untrained baseline

import os
import time
import glob

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from env import ClashRoyaleEnv
from takeAction import get_action_mask
from getState import createFinalState


TURN_TIME = 0.25
WAIT_AFTER_TURN = 0
CHECKPOINT_DIR = "checkpoints"


def load_latest_checkpoint():
    """Load the most recent left checkpoint."""
    files = sorted(glob.glob(f"{CHECKPOINT_DIR}/left_*.zip"), key=os.path.getmtime)

    if not files:
        print("ERROR: No checkpoints found")
        return None, 0

    latest = files[-1]
    ep = int(latest.split("_")[1].split(".")[0])

    print(f"Loading checkpoint from train_iter {ep}: {latest}")
    return latest, ep


def reset_battle(env_left, env_right):
    """Reset and start a new battle."""
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


def main():
    # Initialize environments
    env_left = ClashRoyaleEnv("left")
    env_right = ClashRoyaleEnv("right")

    # Load trained checkpoint for left
    checkpoint_path, train_iter = load_latest_checkpoint()
    if checkpoint_path is None:
        return

    model_left = MaskablePPO.load(checkpoint_path)
    model_left.set_env(env_left)
    print("✓ Loaded TRAINED model (left)")

    # Create fresh untrained model for right
    model_right = MaskablePPO(MaskableActorCriticPolicy, env_right, verbose=0)
    print("✓ Created UNTRAINED model (right)")

    print("\n" + "="*60)
    print("EVALUATION: Trained (left) vs Untrained (right)")
    print("="*60 + "\n")

    # Reset battle
    print("Resetting battle...")
    obs_left, info_left, obs_right, info_right = reset_battle(env_left, env_right)
    time.sleep(1)

    reward_left_sum = 0.0
    reward_right_sum = 0.0
    left_buf = []
    right_buf = []

    print("Starting evaluation game...\n")

    # Play one match
    turn_count = 0
    while True:
        # LEFT TURN (trained)
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

        turn_count += 1
        if turn_count % 10 == 0:
            print(f"Turn {turn_count}: Left={reward_left_sum:.1f}, Right={reward_right_sum:.1f}")

        if done_left:
            winner = info_left.get('winner')
            print(f"\n{'='*60}")
            print(f"GAME OVER - Winner: {winner}")
            print(f"{'='*60}")
            createFinalState(info_left, info_right, obs_left, obs_right, left_buf, right_buf, True, False)
            break

        # RIGHT TURN (untrained)
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
            winner = info_right.get('winner')
            print(f"\n{'='*60}")
            print(f"GAME OVER - Winner: {winner}")
            print(f"{'='*60}")
            createFinalState(info_left, info_right, obs_left, obs_right, left_buf, right_buf, False, True)
            break

    # Print final results
    print(f"\nFinal Results:")
    print(f"  Total turns: {turn_count}")
    
    if winner == "self":
        trained_won = info_left.get('winner') == 'self' if done_left else info_right.get('winner') == 'opp'
        if trained_won:
            print("\n✓ TRAINED model WON!")
        else:
            print("\n✗ UNTRAINED model won")
    else:
        print(f"\n  Winner: {winner}")


if __name__ == "__main__":
    main()