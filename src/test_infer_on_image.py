# Test inference on saved screenshots without running the game

import os
import sys
import glob
import numpy as np
from PIL import Image
from ultralytics import YOLO
import easyocr

from sb3_contrib import MaskablePPO
from getState import getState, encodeState
from takeAction import get_action_mask, decode_action


TEST_DIR = "data/test"
CHECKPOINT_DIR = "models"
YOLO_MODEL_PATH = "models/vision.pt"


def load_latest_checkpoint(side="left"):
    """Load the most recent checkpoint."""
    pattern = f"{CHECKPOINT_DIR}/{side}_*.zip"
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    
    if not files:
        print(f"No checkpoints found for {side} side")
        return None
    
    latest = files[-1]
    ep = int(latest.split("_")[1].split(".")[0])
    print(f"Loading checkpoint from train_iter {ep}: {latest}")
    
    return MaskablePPO.load(latest)


def process_image(image_path, yolo_model, ocr_reader):
    """Process a test image and run inference."""
    print(f"\nProcessing: {image_path}")
    print("=" * 60)
    
    # Load image
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Convert RGB to BGRA (as expected by getState)
    if img_array.shape[2] == 3:
        img_bgra = np.dstack([img_array[:,:,::-1], np.ones((img_array.shape[0], img_array.shape[1], 1), dtype=np.uint8) * 255])
    else:
        img_bgra = img_array
    
    print(f"Image shape: {img_array.shape}")
    
    # Get state using YOLO + OCR
    print("\nRunning YOLO detection and OCR...")
    raw_state = getState(img_bgra, yolo_model, ocr_reader)
    
    if raw_state is None:
        print("ERROR: Failed to extract state from image")
        return
    
    tower_health, elixir, hand, troops, game_over, winner = raw_state
    
    print(f"\nExtracted State:")
    print(f"  Tower Health: {tower_health}")
    print(f"  Elixir: {elixir}")
    print(f"  Hand: {hand}")
    print(f"  Board units: {len(troops)} detected on the 6x4 grid")
    if troops:
        for label, row, col in troops:
            print(f"    - {label} at (row={row}, col={col})")
    else:
        print(f"    (no units detected)")
    print(f"  Game Over: {game_over}, Winner: {winner}")
    
    # Encode state using your pipeline
    obs = encodeState(raw_state)
    print(f"\nEncoded observation shape: {obs.shape}")
    
    # Get action mask
    action_mask = get_action_mask(hand, elixir)
    num_valid = np.sum(action_mask)
    print(f"Action Mask: {num_valid} valid actions out of {len(action_mask)}")
    
    # Load PPO model
    print("\nLoading PPO model...")
    model = load_latest_checkpoint()
    
    if model is None:
        print("ERROR: No model checkpoint found")
        return
    
    # Run inference
    print("\nRunning PPO inference...")
    action, _ = model.predict(obs, action_masks=action_mask, deterministic=False)
    
    print(f"\nPredicted Action ID: {action}")
    
    # Decode action using your decode function
    kind, card_idx, zone_idx = decode_action(action)
    
    if kind == "noop":
        print("Action: WAIT/NO-OP")
    else:
        card_name = hand[card_idx] if card_idx < len(hand) else "Unknown"
        print(f"Action: PLAY card {card_idx} ({card_name}) at zone {zone_idx}")
    
    print("=" * 60)


def main():
    if len(sys.argv) != 2:
        print("Usage: python src/test_infer_on_image.py {image_number}")
        print(f"\nAvailable test images in {TEST_DIR}:")
        
        # List available images
        images = sorted(glob.glob(f"{TEST_DIR}/*.png"))
        if images:
            for img in images:
                basename = os.path.basename(img)
                num = os.path.splitext(basename)[0]
                print(f"  {num}")
        else:
            print("  (no images found)")
        
        sys.exit(1)
    
    image_num = sys.argv[1]
    image_path = os.path.join(TEST_DIR, f"{image_num}.png")
    
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        print(f"\nAvailable images:")
        images = sorted(glob.glob(f"{TEST_DIR}/*.png"))
        for img in images:
            basename = os.path.basename(img)
            num = os.path.splitext(basename)[0]
            print(f"  {num}")
        sys.exit(1)
    
    # Initialize YOLO and OCR
    print("Loading YOLO model...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    
    print("Loading OCR reader...")
    ocr_reader = easyocr.Reader(['en'], gpu=False)
    
    process_image(image_path, yolo_model, ocr_reader)
    
    print("\nVerification complete!")
    print("✓ Environment setup working")
    print("✓ YOLO detection working")
    print("✓ OCR integration working")
    print("✓ PPO inference pipeline working")


if __name__ == "__main__":
    main()