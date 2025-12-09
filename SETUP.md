# Project Setup Guide

This document provides complete installation and setup instructions for running the Clash Royale Hog 2.6 Reinforcement Learning Bot.  
Because the project interacts with the actual Clash Royale game client and depends on precise screen capture positions, some manual configuration is required.

---

## 1. System Requirements

- macOS or Windows machine capable of running an Android emulator  
- Python **3.10+**  
- Conda (recommended)  
- Enough CPU to run YOLOv8 inference in real time  
- BlueStacks or another Android emulator  
- Two Clash Royale accounts (for self-play data collection)

> **Note for graders:**  
> You do *not* need to run actual Clash Royale to test the code. A mock screenshot test is included.

---

## 2. Install Dependencies

### 2.1 Clone the repository

git clone <REPO_URL>
cd <REPO_NAME>

### 2.2 Create the environment

conda env create -f environment.yml
conda activate clashroyale

## 3. Emulator setup

> **Graders:**  
> You do not need to configure BlueStacks unless you want to run inference. You can skip to section 7 to test without the game.

If you want to run the actual bot:
1. Install Bluestacks (or another Android emulator)
2. Download and open Clash Royale in the emulator
3. Resize and position the emulator window so that the game arena appears exactly within the expected region-of-interest (ROI)

The project expects two different ROIs for:
- Left-side agent window
- Right-side agent window
(used during self-play)

## 4. Verify ROI Alignment
Run python src/setup_test.py.

This script captures the ROI defined in vars.py and displays a screenshot.
Compare the output against the reference image provided in the README.

If the screenshot does not match, resize and move the emulator window until the screenshot matches. 
This step is critical; YOLO and pixel-based detection depend on exact alignment.

## 5. Running the Bot

### 6.1 Inference
Start a match manually in Clash Royale, then run python src/infer.py.

The agent will:
- Capture frames
- Extract game state
- Predict actions using PPO
- Execute mouse clicks automatically
No gameplay data is recorded during inference.

### 6.2 Collecting Self-play data (optional)
Requires:
- Two emulator windows
- Two Clash Royale accounts logged in
- The accounts must be friends

Steps:
1. Position both emulator windows according to LEFT_ROI and RIGHT_ROI
2. Navigate both to the Friendly Battle screen
3. Adjust NEXT_EPISODE and TOTAL_EPISODES in collect_data.py
4. Run: python src/collect_data.py

Episodes will be saved into: buffers/ep_*.json

**Warning:**
OCR sometimes misreads the “game over” screen.
During long unattended runs, the agent may misclick after the match ends.

### 7. Testing without runnning Clash Royale (for graders)
To allow testing on systems without an emulator or the actual game:
1. Test screenshots are stored in data/test
2. run python src/test_infer_on_image.py {image_number}

This script:
- Loads the YOLO model
- Processes the test image
- Runs the PPO policy
- Prints the predicted action

This verifies:
- Environment setup
- YOLO detection
- OCR integration
- PPO inference pipeline
without needing to run the real game.

### 8. Training the PPO model
After collecting episodes, run python src/train.py
This script:
- Loads your replay buffers
- Builds a ReplayEnv
- Runs offline PPO with action masking
- Saves checkpoints into checkpoints/
- Stops early using a patience-based convergence test
No emulator or game interaction is required for training.