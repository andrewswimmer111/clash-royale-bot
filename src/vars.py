# STATE VARS

LEFT_ARENA_ROI = {"top": 71, "left": 0, "width": 509, "height": 911}
RIGHT_ARENA_ROI = {"top": 72, "left": 967, "width": 509, "height": 911}

# Else is formatted (x1, y1, x2, y2)
PLAYABLE_ARENA = (35, 75, 474, 700)

TOWER_HEALTHBARS = [
    (103, 138, 157, 139),  # top-left tower
    (103, 566, 157, 567),  # bottom-left tower
    (374, 566, 426, 567),  # bottom-right tower
    (374, 138, 426, 139)   # top-right tower
]

CARD_SLOTS = [
    (113, 754, 203, 869),
    (209, 754, 299, 869),
    (305, 754, 395, 869),
    (403, 754, 493, 869),
]
ELIXIR_ROI = (140, 886, 490, 896)

GAME_OVER = (190, 800, 320, 850)
WINNER_OPP = (180, 80, 320, 120)
WINNER_SELF = (185, 350, 325, 390)

ALL_CARDS = ["cannon", "fireball", "hog", "ice_golem", "ice_spirit", "log", "musk", "skeleton", "empty"]
CARD_TO_ID = {name: idx for idx, name in enumerate(ALL_CARDS)}

TROOP_CARDS = [c for c in ALL_CARDS if c != "empty"]
ALL_TROOPS = [f"A_{card}" for card in TROOP_CARDS] + \
             [f"E_{card}" for card in TROOP_CARDS] + \
             ["A_L_tower", "A_R_tower", "E_L_tower", "E_R_tower"]
TROOP_TO_ID = {name: idx for idx, name in enumerate(ALL_TROOPS)}

# ACTION VARS
ACTION_ZONES = {
    1: (71, 406),   # Left bridge 1
    2: (96, 406),   # Left bridge 2
    3: (121, 406),  # Left bridge 3
    4: (241, 406),  # Center bridge 1
    5: (266, 406),  # Center bridge 2
    6: (387, 406),  # Right bridge 1
    7: (412, 406),  # Right bridge 2
    8: (437, 406),  # Right bridge 3

    9: (71, 530),   # Far left defense
    10: (437, 530), # Far right defense

    11: (121, 492), # Left lane defense 1
    12: (121, 506), # Left lane defense 2
    13: (121, 530), # Left lane defense 3
    14: (387, 492), # Right lane defense 1
    15: (387, 506), # Right lane defense 2
    16: (387, 530), # Right lane defense 3

    17: (241, 571), # Left middle deep
    18: (266, 571), # Right middle deep

    19: (241, 492), # Left middle high
    20: (241, 506), # Left middle low
    21: (266, 492), # Right middle high
    22: (266, 506), # Right middle low

    23: (191, 241), # Fireball left
    24: (310, 241)  # Fireball right
}

CARD_ALLOWED_PLACEMENTS = {
    "cannon": [9, 10, 19, 20, 21, 22],
    "empty": [],
    "fireball": [3, 6, 23, 24],
    "hog": list(range(1, 9)),
    "ice_golem": list(range(1, 23)),
    "ice_spirit": list(range(1, 23)),
    "log": [3, 4, 5, 6, 13, 16],
    "musk": [1, 4, 5, 8, 9, 10, 17, 18],
    "skeleton": list(range(1, 23))
}

CARD_COSTS = {
    "cannon": 3,
    "empty": 0,
    "fireball": 4,
    "hog": 4,
    "ice_golem": 2,
    "ice_spirit": 1,
    "log": 2,
    "musk": 4,
    "skeleton": 1
}
CARD_SLOT_POINTS = [
    (158, 811),
    (254, 811),
    (350, 811),
    (448, 811),
]
TOWER_NAMES = ["E_L_tower", "A_L_tower", "A_R_tower", "E_R_tower"]

# SELF_PLAY VARS
START_CENTERS = {
    "GAME_OVER": (240, 825),
    "FRIEND": (328, 95),
    "OPP": (378, 313),
    "FRIENDLY_BATTLE": (403, 385),
    "INVITE": (378, 263),
    "ACCEPT": (303, 275)
}
GAME_OVER_CLICK = (240, 825)

FOCUS_RIGHT = (480, 400)
FOCUS_LEFT = (50, 400)