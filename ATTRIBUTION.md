# ATTRIBUTION

This document provides detailed attribution for all external code, libraries, datasets, model weights, and AI-assisted development used in this project.

---

## 1. AI-Assisted Code Generation

Portions of this project were developed with assistance from **ChatGPT (OpenAI GPT-4.1 and GPT-5.1 models)**.  
AI assistance was used specifically for:

- Designing the *ReplayEnv* architecture for offline PPO training  
- Debugging errors involving MaskablePPO and action masking  
- Refactoring training loops and improving efficiency  
- Drafting documentation (README, SETUP.md, ATTRIBUTION.md)  
- Clarifying PPO behavior and hyperparameter interpretation  
- Generating helper functions and logic scaffolding  
- Suggesting code structure and best practices for offline RL pipelines  

All AI-generated code was reviewed, tested, and modified by the author before inclusion.

---

## 2. External Libraries and Frameworks

This project relies on the following open-source libraries:

### **Reinforcement Learning**
- **Stable-Baselines3**  
  - License: MIT  
  - Repository: https://github.com/DLR-RM/stable-baselines3  
  - Used for PPO implementation and training infrastructure.

- **SB3-Contrib (MaskablePPO)**  
  - License: MIT  
  - Repository: https://github.com/Stable-Baselines-Team/stable-baselines3-contrib  
  - Used for action masking and improved discrete action space control.

### **Vision & OCR**
- **Ultralytics YOLOv8**  
  - License: AGPL-3.0  
  - Repository: https://github.com/ultralytics/ultralytics  
  - Used for detecting troops, buildings, and spells on screen.

- **easyOCR**  
  - License: Apache-2.0  
  - Repository: https://github.com/JaidedAI/EasyOCR  
  - Used for episode termination detection ("OK", "Winner" text).

### **Screen Capture & Automation**
- **mss**  
  - License: MIT  
  - Used for capturing real-time screenshots of the emulator window.

- **pyautogui**  
  - License: BSD  
  - Used for clicking, dragging, and simulating card placements.

### **General**
- **NumPy** (BSD-3)  
- **OpenCV** (Apache-2.0)  
- **PyTorch** (BSD-style license)  
- **Gymnasium** (MIT)

---

## 3. Pretrained and Fine-Tuned Models

### **YOLOv8 Model (vision.pt)**
- Base model provided by Ultralytics under AGPL-3.0  
- Fine-tuned by the project author using a custom dataset of Clash Royale troop images.  
- Portions of the dataset were collected manually by the author.

### **OCR Weights**
- Downloaded automatically by easyOCR under Apache-2.0 licensing.

### **PPO Checkpoints (left_*.zip, right_*.zip)**
- Produced entirely by this project using self-play data.
- No external reinforcement learning models were used or imported.

---

## 4. Datasets

### **Clash Royale Detection Dataset**
- Created manually by the author using screenshots captured via BlueStacks.
- Labeled using Roboflow
- Dataset links:
    - Original: https://app.roboflow.com/andrewswimmer111/hog-2-6-zfi09/browse
    - Fine-tuning: https://app.roboflow.com/andrewswimmer111/hog-2-6-new-arena-ey3ta/browse
- Dataset documentation:
    - I took screenshots every 1 second while playing a few games of clash with my friend.
    - Uploaded the data into roboflow and manually labeled some
    - Trained a vision model in roboflow to assist with labelling
    - Went through and confirmed/corrected all remaining images

### **YOLO Training Augmentations**
Provided by the Ultralytics training pipeline.

---

## 5. External Software

- **Clash Royale (Supercell)**  
  - Game client used solely to capture screenshots for academic research.  
  - No copyrighted assets are redistributed in this repository.  
  - This project is not affiliated with, endorsed, or sponsored by Supercell.

- **BlueStacks Emulator**  
  - Used to run Clash Royale on macOS for screen capture and automation.  
  - Not included in this repository.

---

## 6. Algorithmic Attribution

This project uses standard RL algorithms:

- **Proximal Policy Optimization (PPO)**  
  - Schulman et al., 2017  
  - https://arxiv.org/abs/1707.06347  

- **Action Masking**  
  - Implemented using SB3-Contrib (MaskablePPO)  


## 7. Summary

This project stands on the shoulders of several important open-source tools (YOLOv8, Stable-Baselines3, PyTorch, easyOCR).  
AI assistance was used for structuring the RL training loop, debugging mask logic, and improving documentation, but all final engineering decisions and implementation were performed by the author.

