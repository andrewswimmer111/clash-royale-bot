from vars import PLAYABLE_ARENA, TOWER_HEALTHBARS, CARD_SLOTS, ELIXIR_ROI, GAME_OVER, WINNER_OPP, WINNER_SELF, CARD_TO_ID, TROOP_TO_ID, TOWER_NAMES

import cv2
import os
import re
import numpy as np
from difflib import SequenceMatcher

def getTowerHealth(frame):
    '''
    Input: a frame
    Output: [top-left, bottom-left, bottom-right, top-right] tower health respectively. 
    '''
    health_values = []
    for (x1, y1, x2, y2) in TOWER_HEALTHBARS:
        bar = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(bar, cv2.COLOR_BGR2HSV)
        v = hsv[:,:,2]

        _, width = v.shape
        segment_width = width // 5

        health = 0
        for i in range(5):
            sx = i * segment_width
            seg_v = v[:, sx:sx+segment_width]

            mean_v = seg_v.mean()   # mean brightness of this segment
            if mean_v > 150:
                health += 1
            else:
                break
        health_values.append(health / 5) # Normalize to between (0 and 1)

    return health_values

def getElixir(frame):
    x1, y1, x2, y2 = ELIXIR_ROI
    elixir = 0
    for i in range(1, 11):
        leftmost_x = (x1 - 15) + (i * 35)
        square = frame[y1:y2, leftmost_x:leftmost_x+5]

        hsv = cv2.cvtColor(square, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        mean_h, mean_s, mean_v = h.mean(), s.mean(), v.mean()

        if 125 <= mean_h <= 165 and mean_s > 60 and mean_v > 40:
            # Meaning the box is purple
            elixir = i
    return elixir

def loadTemplates(path):
    '''
    Loads image templates
    '''
    templates = {}
    for filename in os.listdir(path):
        if filename.endswith(".png"):
            name = os.path.splitext(filename)[0]
            img = cv2.imread(os.path.join(path, filename))
            templates[name] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return templates

def identifyCard(slot_img_gray, templates):
    '''
    Identifies which card is in each slot
    '''
    best_match = None
    best_score = 0
    for name, template in templates.items():
        if name == "game_over": continue
        result = cv2.matchTemplate(slot_img_gray, template, cv2.TM_CCOEFF_NORMED)
        score = result.max()
        if score > best_score:
            best_score = score
            best_match = name
    return best_match

def getHand(frame, templates):
    '''
    Identifies what cards are in the 4 slots
    '''
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cards_in_hand = []
    for (x1, y1, x2, y2) in CARD_SLOTS:
        slot_img = gray[y1:y2, x1:x2]
        name = identifyCard(slot_img, templates)
        cards_in_hand.append(name)
    return cards_in_hand


def classifyUniformGrid(cx, cy, arena = PLAYABLE_ARENA, rows=6, cols=4):
    """
    cx, cy: center of the unit (absolute pixel coords in arena crop)
    arena: playable arena(x1, y1, x2, y2)
    rows, cols: grid dimensions (uniform)
    Returns: (row, col) indices
    """
    x1, y1, x2, y2 = arena

    W = x2 - x1
    H = y2 - y1
    # Convert to relative [0,1]
    rx = (cx - x1) / W
    ry = (cy - y1) / H

    # Uniform index
    col = int(rx * cols)
    row = int(ry * rows)

    # clamp
    col = min(max(col, 0), cols - 1)
    row = min(max(row, 0), rows - 1)

    return (row, col)

def getTroopsOnBoard(image, model, conf=0.5):
    '''
    Input: 
        Frame, vision model, confidence threshold
    Output: 
        List of (troop, row, col) - (The board is split into 6 rows and 4 columns)
    '''
    results = model.predict(source=image, conf=conf, verbose=False)
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        row, col = classifyUniformGrid(cx, cy)

        detections.append((label, row, col))
    return detections

def get_text_from_frame(frame, ocr):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    gray = cv2.equalizeHist(gray)
    
    h, w = gray.shape
    gray = cv2.resize(gray, (w * 3, h * 3), interpolation=cv2.INTER_LINEAR)

    results = ocr.readtext(gray, detail=0, paragraph=False)
    text = ''.join(results)
    tokens = re.findall(r'[A-Za-z0-9]+', text)

    return "".join(tokens)

def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def checkGameOver(frame, templates, troops, ocr):
    x1, y1, x2, y2 = GAME_OVER
    button_crop = frame[y1:y2, x1:x2]
    button_text = get_text_from_frame(button_crop, ocr)
    button_score = similarity(button_text, "okqr")  # sometimes misreads "ok" as "qr"

    if troops == [] and button_score > 0.01:        # low threshold is acceptable: essentially never exceeds 0.0 unless the button is there

        x1, y1, x2, y2 = WINNER_OPP
        opp_crop = frame[y1:y2, x1:x2]
        opp_text = get_text_from_frame(opp_crop, ocr)

        x1, y1, x2, y2 = WINNER_SELF
        self_crop = frame[y1:y2, x1:x2]
        self_text = get_text_from_frame(self_crop, ocr)

        opp_score = similarity(opp_text, "winner")
        self_score = similarity(self_text, "winner")

        if opp_score > self_score and opp_score > 0.5:
            return True, "opp"
        elif self_score > opp_score and self_score > 0.5:
            return True, "self"
        else:
            print("Winner unclear")
            return True, "opp"
    else:
        return False, None

def getState(frame, model, ocr):
    '''
    Takes in a BGRA frame of a game. 
    Returns [tower health, elixir, hand, troops, game_over, opp_win, self_win]
    Returns 
    '''
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    templates = loadTemplates('data/templates')

    tower_health = getTowerHealth(frame)
    elixir = getElixir(frame)
    hand = getHand(frame, templates)
    troops = getTroopsOnBoard(frame, model)

    game_over, winner = checkGameOver(frame, templates, troops, ocr)

    res = [tower_health, elixir, hand, troops, game_over, winner]

    return res

def encodeState(state):
    '''
    Takes in [tower health, elixir, hand, troops, game_over, opp_win, self_win]
    Encodes it for Gymnasium ENV
    '''

    tower_health, elixir, hand, troops, game_over, winner = state

    # Mask tower health for destroyed towers
    tower_set = {label for (label, *_rest) in troops}

    masked_health = []
    for idx, tower_label in enumerate(TOWER_NAMES):
        if tower_label in tower_set:
            masked_health.append(tower_health[idx])
        else:
            masked_health.append(0.0)  # mask health when tower not detected

    masked_health = np.array(masked_health, dtype=np.float32)

    # Normalize elixir
    elixir_enc = np.array([elixir / 10.0], dtype=np.float32)

    # Encode hand
    onehot = np.zeros((4, 9), dtype=np.float32)
    for i, card in enumerate(hand):
        cid = CARD_TO_ID[card]
        onehot[i, cid] = 1.0
    hand_enc = onehot.flatten()

    # Encode troops
    troops_enc = np.zeros((20, 6, 4), dtype=np.float32)
    for label, row, col in troops:
        troop_id = TROOP_TO_ID[label]
        troops_enc[troop_id, row, col] += 1  # allows stacking
    troops_enc = troops_enc.flatten()  # 480 dims (20 x 24)

    # Game flags
    game_over_f = float(game_over)
    win_self = 1.0 if winner == "self" else 0.0
    win_opp = 1.0 if winner == "opp" else 0.0

    flag_enc = np.array([game_over_f, win_self, win_opp], dtype=np.float32)

    obs = np.concatenate([
        masked_health,   # 4
        elixir_enc,      # 1
        hand_enc,        # 36
        troops_enc,      # 480
        flag_enc         # 3
    ])

    # Size 524
    return obs

def createFinalState(
    info_left, info_right,
    obs_left, obs_right,
    left_buf, right_buf,
    done_left=False, done_right=False
):
    if done_left:
        winner = info_left.get("winner")

        if winner == "self":        # Left won
            r_left_terminal = +50
            r_right_terminal = -50
        else:                       # Right won
            r_left_terminal = -50
            r_right_terminal = +50

        # Add terminal reward to LEFT's final transition
        left_buf[-1]["reward"] += r_left_terminal

        # Append dummy final transition for RIGHT
        right_buf.append({
            "obs": obs_right,
            "action": None,
            "reward": r_right_terminal,
            "next_obs": obs_right,
            "done": True,
            "info": {"action_mask": info_right["action_mask"]},
        })

    elif done_right:
        winner = info_right.get("winner")

        if winner == "self":        # Right won
            r_left_terminal = -50
            r_right_terminal = +50
        else:                       # Left won
            r_left_terminal = +50
            r_right_terminal = -50

        # Add terminal reward to RIGHT's final transition
        right_buf[-1]["reward"] += r_right_terminal

        # Append dummy final transition for LEFT
        left_buf.append({
            "obs": obs_left,
            "action": None,
            "reward": r_left_terminal,
            "next_obs": obs_left,
            "done": True,
            "info": {"action_mask": info_left["action_mask"]},
        })

    # No need to return — buffers mutate in place
