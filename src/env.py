from vars import LEFT_ARENA_ROI, RIGHT_ARENA_ROI, START_CENTERS, GAME_OVER_CLICK, FOCUS_RIGHT, FOCUS_LEFT
from takeAction import mouse_click, play_action, get_action_mask
from getState import getState, encodeState

import gymnasium as gym
from gymnasium import spaces
from ultralytics import YOLO
import easyocr

import numpy as np
import pyautogui
import time
import mss
import threading


VISION_MODEL = YOLO("models/vision.pt")
OCR_ENGINE = easyocr.Reader(['en'], gpu=False)

# Allows program to interact with the game
class ClashRoyaleEnv(gym.Env):
    def __init__(self, side):
        '''
        role: initiator / acceptor
        side: left / right
        '''
        # Unique roles
        self.role = "initiator" if side == "left" else "acceptor"
        self.arena_roi = LEFT_ARENA_ROI if side == "left" else RIGHT_ARENA_ROI
        self.focus = FOCUS_LEFT if side == "left" else FOCUS_RIGHT

        # Individual models
        self.vision_model = VISION_MODEL
        self.ocr = OCR_ENGINE

        # Observation space: 4 + 1 + 36 + 480 + 3 = 524 dims
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(524,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(97)

        # Others
        self.raw_state = None
        self.prev_raw_state = None
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Just reset internal state
        self.prev_raw_state = None
        self.raw_state = None

        # Wait for the trainer to initiate battle flow
        # Trainer will call _start_battle() / _accept_battle() manually

        # Return a dummy observation until the trainer replaces it
        dummy = np.zeros((524,), dtype=np.float32)
        info = {"action_mask": np.zeros(97, dtype=np.int8)}
        return dummy, info

    
    def _get_obs(self):
        with mss.mss() as sct:
            frame = np.array(sct.grab(self.arena_roi))
            raw_state = getState(frame, self.vision_model, self.ocr)
            obs = encodeState(raw_state)

            self.prev_raw_state = getattr(self, "raw_state", raw_state)
            self.raw_state = raw_state
        return obs
    
    def step(self, action):

        # Execute action
        play_action(action, self.arena_roi, self.focus)
        time.sleep(0.3)

        # Observe new state
        next_obs = self._get_obs()

        # Game over?
        game_over, winner = self.raw_state[4], self.raw_state[5]
        done = bool(game_over)

        # Compute reward
        reward = self._compute_reward(self.prev_raw_state, self.raw_state)

        # Action mask for next frame
        next_hand = self.raw_state[2]
        next_elixir = self.raw_state[1]
        action_mask = get_action_mask(next_hand, next_elixir)

        info = {"action_mask": action_mask, "winner": winner}

        return next_obs, reward, done, False, info

    def _compute_reward(self, prev_state, next_state):

        # Beause we want to win, winning reward takes up most (+50)
        # but we also want to reward good mechanics, so holding elixir, not leaking, damaging towers
        # Rewards are not based off troops bc of the unreliability of YOLO accuracy
        
        reward = 0.0

        if prev_state is not None:

            # Tower damage rewards
            prev_hp = np.array(prev_state[0], dtype=float)
            next_hp = np.array(next_state[0], dtype=float)
            
            prev_enemy = prev_hp[[0, 3]].sum()
            next_enemy = next_hp[[0, 3]].sum()
            prev_ally = prev_hp[[1, 2]].sum()
            next_ally = next_hp[[1, 2]].sum()

            reward += (prev_enemy - next_enemy) - (prev_ally - next_ally)

            # Elixir spending rewards
            prev_elixir = float(prev_state[1])
            next_elixir = float(next_state[1])

            # Gain slight reward for getting more elixir
            reward += 0.01 * (next_elixir - prev_elixir)

            if prev_elixir == 10 and next_elixir == 10:
                reward -= 0.05

        # Game outcome rewards (handled in training)
        game_over = next_state[4]
        winner = next_state[5]

        if game_over:
            if winner == "self":
                reward += 0
            elif winner == "opp":
                reward -= 0
        return float(reward)


    def _click_point(self, point, delay=0.5):
        """Clicks a (x, y) point relative to arena ROI."""
        x, y = point
        mouse_click(self.arena_roi, x, y)
        time.sleep(delay)
    
    def _start_battle(self):
        print("starting battle")
        self._click_point(self.focus, 0.5)
        self._click_point(GAME_OVER_CLICK, 2.0)

        for key in ["FRIEND", "OPP", "FRIENDLY_BATTLE", "INVITE"]:
            self._click_point(START_CENTERS[key], 0.5)

        time.sleep(2)
        
    def _accept_battle(self):
        print("accepting battle")
        self._click_point(self.focus, 0.5)
        self._click_point(GAME_OVER_CLICK, 2.0)

        for key in ["FRIEND", "ACCEPT"]:
            self._click_point(START_CENTERS[key], 0.5)

        time.sleep(2)

# Allows imported ppo agent to train since step can't be called like that in ClashEnv
class ReplayEnv(gym.Env):
    def __init__(self, transitions):
        super().__init__()

        self.transitions = transitions
        self.pos = 0

        # infer obs and action shapes from the first transition
        first = transitions[0]

        obs_shape = first["obs"].shape
        action_mask = first["info"]["action_mask"]
        n_actions = len(action_mask)

        # Gym spaces (needed by SB3) -- identical to ClashEnv
        self.observation_space = spaces.Box(low=-1, high=1, shape=obs_shape, dtype=np.float32)
        self.action_space = spaces.Discrete(n_actions)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.pos = 0
        tr = self.transitions[0]
        return tr["obs"], tr["info"]

    def step(self, action):
        tr = self.transitions[self.pos]
        self.pos += 1
        return (
            tr["next_obs"],
            tr["reward"],
            tr["done"],
            False,
            tr["info"]
        )

    def render(self): pass
    def close(self): pass