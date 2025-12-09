# Single agent inference - runs continuously without storing buffers

import os
import sys
import time
import glob

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from env import ClashRoyaleEnv
from takeAction import get_action_mask


INFERENCE_INTERVAL = 1
MODEL_DIR = "models"


def load_latest_checkpoint(side):
    """Load the most recent checkpoint for the given side."""
    pattern = f"{MODEL_DIR}/{side}_*.zip"
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    
    if not files:
        print(f"No checkpoints found for {side} side - cannot run inference")
        return None
    
    latest = files[-1]
    ep = int(latest.split("_")[1].split(".")[0])
    print(f"Loading {side} checkpoint from train_iter {ep}: {latest}")
    
    return latest


def main():
    if len(sys.argv) != 3 or sys.argv[1].upper() not in ["L", "R"] or sys.argv[2].upper() not in ["L", "R"]:
        print("Usage: python3 src/infer.py {L/R} {L/R}")
        print("  First argument: arena_roi side (L/R)")
        print("  Second argument: checkpoint side (L/R)")
        sys.exit(1)
    
    roi_arg = sys.argv[1].upper()
    checkpoint_arg = sys.argv[2].upper()
    
    roi_side = "left" if roi_arg == "L" else "right"
    checkpoint_side = "left" if checkpoint_arg == "L" else "right"
    
    print(f"Starting inference:")
    print(f"  Arena ROI: {roi_side}")
    print(f"  Checkpoint: {checkpoint_side}")
    
    # Load environment with roi_side
    env = ClashRoyaleEnv(roi_side)
    
    # Load latest checkpoint from checkpoint_side
    checkpoint_path = load_latest_checkpoint(checkpoint_side)
    if checkpoint_path is None:
        sys.exit(1)
    
    model = MaskablePPO.load(checkpoint_path)
    model.set_env(env)
    
    # Reset environment
    obs, info = env.reset()

    
    print(f"Running continuous inference (interval: {INFERENCE_INTERVAL}s)")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            # Get action mask
            mask = info["action_mask"]
            
            # Predict action
            action, _ = model.predict(obs, action_masks=mask, deterministic=False)
            
            # Take step
            t0 = time.time()
            obs, reward, done, _, info = env.step(action)
            
            # Sleep to maintain interval
            elapsed = time.time() - t0
            if elapsed < INFERENCE_INTERVAL:
                time.sleep(INFERENCE_INTERVAL - elapsed)
            
            # Reset if done
            if done:
                print(f"Episode finished - quitting")
                obs = env.reset()
                break
    
    except KeyboardInterrupt:
        print("\nInference stopped by user")


if __name__ == "__main__":
    main()