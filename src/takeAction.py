from vars import ACTION_ZONES, CARD_SLOT_POINTS, CARD_ALLOWED_PLACEMENTS, CARD_COSTS, FOCUS_LEFT, FOCUS_RIGHT

import numpy as np
import time
import pyautogui

def mouse_click(arena_roi, x, y, delay=0.02):
    """
    Clicks at (x, y) relative to the arena ROI.
    (x, y) are coordinates inside the cropped arena frame.
    """
    abs_x = arena_roi["left"] + x
    abs_y = arena_roi["top"] + y

    pyautogui.moveTo(abs_x, abs_y)
    time.sleep(delay)
    pyautogui.mouseDown()
    time.sleep(delay)
    pyautogui.mouseUp()


def decode_action(action_id, num_zones=24):
    if action_id == 0:
        return ("noop", -1, -1)

    action_id -= 1  # remove noop offset

    card_idx = action_id // num_zones           # 0-index into hand
    zone_idx = (action_id % num_zones) + 1      # 1-index into zone dict

    return ("play", card_idx, zone_idx)

def get_action_mask(hand, elixir):
    mask = np.zeros(97, dtype=bool) # 97 possible actions (1 + 4(24))

    # Always allow "do nothing"
    mask[0] = True

    for card_idx, card_name in enumerate(hand):
        allowed_zones = CARD_ALLOWED_PLACEMENTS[card_name]
        cost = CARD_COSTS[card_name]
        for zone in allowed_zones:
            action_id = 1 + card_idx * 24 + (zone - 1) # since zone is 1 indexed
            if elixir >= cost:
                mask[action_id] = True
    return mask

def play_action(action_id, arena_roi, focus):
    kind, card_idx, zone_idx = decode_action(action_id)

    if kind == "noop":
        return
    
    # Focus on window
    x, y = focus
    mouse_click(arena_roi, x, y)
    
    # click card slot
    card_slot_x, card_slot_y = CARD_SLOT_POINTS[card_idx]
    mouse_click(arena_roi, card_slot_x, card_slot_y)

    # click zone
    x, y = ACTION_ZONES[zone_idx]
    mouse_click(arena_roi, x, y)