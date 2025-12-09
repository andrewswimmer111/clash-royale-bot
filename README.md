# Clash Royale Bot - Hog 2.6

This project implements a reinforcement learning agent that plays Clash Royale in a controlled mirror matchup of the Hog 2.6 deck versus itself. The agent learns by gathering its own gameplay data through real-time self-play, then trains offline using PPO to improve its strategy and decision-making.

## What it does

This project works in several stages. First, a screenshot of a specific region of the computer screen is captured during gameplay. That image is passed into a custom fine-tuned YOLOv8 model, which detects all units and identifies which team they belong to.

In addition to card placements, the system also extracts tower health, elixir count, and the player’s current hand. These are obtained either through direct pixel-color inspection at known coordinates or by template-matching segments of the screenshot to pre-recorded images. Episode termination is detected by running an OCR model over predefined UI locations and identifying keywords such as “OK” and “Winner.”

During self-play, two instances of the agent (Left and Right) play against each other in real time. Each agent independently captures its own game state as soon as the other agent finishes its move, ensuring synchronized turn-taking. All recorded transitions from both sides are saved into JSON files inside the buffers/ directory. These files form the replay dataset used for offline training.

During inference (actual bot play), only a single agent is used. It no longer records data; instead, it continuously receives the encoded game state and selects an action based on its trained PPO policy.

Finally, during training, the project loads previously recorded episodes, converts them into a replay environment, and uses offline PPO (with action masking) to update the agent's neural network weights. This process can be repeated iteratively: gather more self-play data, retrain the model, and gradually improve the bot’s performance over time.

## Quick Start

Running this project requires interacting with the actual Clash Royale game client. Since the game cannot be run headlessly, and because the vision pipeline depends on precise pixel locations, some manual setup and window positioning is required.

### 1. Install and Position Clash Royale
To run the bot locally, you must first run Clash Royale on your computer. I used **BlueStacks** (an Android emulator) on a **14-inch MacBook Pro**, but any emulator or mirrored mobile device should work as long as the UI layout is stable.

This project defines two Regions of Interest (ROIs):

- **LEFT_ROI** — created by placing the BlueStacks window on the **left side** of the screen  
- **RIGHT_ROI** — created similarly on the **right side**

If you are running on a different machine, you must re-capture these ROIs.

### 2. Verify Display Alignment

Run: python3 src/setup_test.py

This script captures the ROI and displays a screenshot.

Compare the output to the reference image in this repository.
If the captured screenshot does not match the expected region:

- reposition/resize the BlueStacks window
- ensure no window borders or toolbars overlap the ROI

Once the screenshot matches, the perception pipeline (YOLO + OCR + pixel sampling) will function correctly.

### 3. Running Inference (Playing a Single Game)

1. Start a Clash Royale match manually.  
2. Ensure the BlueStacks window is positioned exactly as required and not covered.  
3. Run: python3 src/infer.py

The agent will begin reading the screen and selecting actions based on its trained PPO policy.
During inference, only one agent is active, and no gameplay data is recorded.

### 4. Running Self-Play (Data Collection)
Self-play requires:

1. Two emulator windows, each logged into separate Clash Royale accounts
2. Both accounts must be friends so they can challenge each other
3. Both windows positioned EXACTLY where the ROIs expect them to be

Once both windows are positioned:
1. Navigate both accounts to the Friendly Battle screen.
2. Adjust the NEXT_EPISODE and TOTAL_EPISODES variables in src/collect_data.py
3. Run python3 src/collect_data.py

The script will begin generating self-play episodes and storing them in the buffers/ directory.

**Warning:**
Due to imperfect OCR detection and timing differences, the agent may occasionally misdetect the “game over” screen, attempt to play a card after the match ends, or click outside intended regions. This behavior messes up the iterative behavior of the script, as the model is no longer on the expected screen. Therefore, self-play collection may require periodic supervision, and likely will not function for long recording sessions.

### 5. Testing the bot (without running Clash Royale)
Because this process requires lots of manual setup, a inference test with pre-recorded screenshots is provided in src/test_infer_on_image.py. This test should provide more detail into how the project pipeline works. 

## Videos

## Evaluation

The current bot’s performance is limited, primarily due to a lack of training data and imperfect perception. The YOLOv8 vision model used to detect troops on the screen was originally trained before a recent Clash Royale update that changed the Friendly Battle arena UI. After discovering this mismatch, I collected a new dataset and fine-tuned the model, but the dataset was small due to the time-consuming nature of manual labeling. As a result, the YOLO model is not consistently accurate, which introduces downstream errors into the reinforcement learning pipeline.

Another major bottleneck is data collection speed. Because Clash Royale cannot be run headlessly, each self-play episode takes roughly **3–6 minutes** to gather. Under time constraints, I was only able to collect around **60 episodes**. In contrast, competitive reinforcement learning systems typically require **thousands to millions** of episodes to develop strong strategic behavior.

Given these constraints, the agent shows movement patterns that are slightly better than random choice, but it is far from competitive or strategically robust. More extensive data collection, improved perception, and continued iterative self-play training would significantly enhance performance.
