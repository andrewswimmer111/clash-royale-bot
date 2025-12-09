import sys
import cv2
import numpy as np
import mss
from vars import (
    LEFT_ARENA_ROI,
    RIGHT_ARENA_ROI,
    TOWER_HEALTHBARS,
    CARD_SLOTS,
    ELIXIR_ROI,
    ACTION_ZONES,
)

# --- Helpers ---
def draw_box(img, box, label, color=(0, 255, 0)):
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def draw_point(img, x, y, label, color=(0, 0, 255)):
    cv2.circle(img, (x, y), 5, color, -1)
    cv2.putText(img, label, (x + 8, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def global_to_local_point(point, roi_left, roi_top):
    """
    Convert a global (screen) point into coordinates relative to the ROI image.
    point = (x, y)
    roi_left, roi_top = top-left corner of the ROI
    """
    px, py = point
    return px - roi_left, py - roi_top

# --- Main ---
def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("L", "R"):
        print("Usage: python3 src/setup_test.py {L|R}")
        sys.exit(1)

    side = sys.argv[1]
    roi = LEFT_ARENA_ROI if side == "L" else RIGHT_ARENA_ROI
    roi_left  = int(roi["left"])
    roi_top   = int(roi["top"])
    roi_w     = int(roi["width"])
    roi_h     = int(roi["height"])

    print(f"[INFO] Capturing {side} ROI: {roi}")

    # Capture ROI screenshot
    with mss.mss() as sct:
        monitor = {"left": roi_left, "top": roi_top, "width": roi_w, "height": roi_h}
        img = np.array(sct.grab(monitor))[:, :, :3].copy()  # drop alpha and make contiguous

    # Tower Healthbars
    for i, (x1, y1, x2, y2) in enumerate(TOWER_HEALTHBARS):
        draw_box(img, (x1, y1, x2, y2), f"TOWER_HB", (0, 255, 255))

    # Card Slots
    for i, (x1, y1, x2, y2) in enumerate(CARD_SLOTS):
        draw_box(img, (x1, y1, x2, y2), f"CARD_SLOT", (255, 0, 0))

    # Elixir ROI
    ex1, ey1, ex2, ey2 = ELIXIR_ROI
    draw_box(img, (ex1, ey1, ex2, ey2), "ELIXIR", (0, 0, 255))

    # Action zones
    for zone_id, (lx, ly) in ACTION_ZONES.items():
        draw_point(img, lx, ly, f"{zone_id}", (0, 255, 0))

    # Save and show
    out_path = f"setup_test_output_{side}.png"
    cv2.imwrite(out_path, img)
    print(f"[INFO] Saved annotated ROI to {out_path}")

    cv2.imshow("Annotated ROI", img)
    print("[INFO] Close the window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()