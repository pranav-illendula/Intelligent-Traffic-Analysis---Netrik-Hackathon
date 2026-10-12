import math

# ================= MEMORY =================
speed_memory = {}
red_jump_memory = set()

def reset_violation_memory():
    global speed_memory, red_jump_memory
    speed_memory = {}
    red_jump_memory = set()

# ================= RASH DRIVING =================
def detect_rash_driving(vehicle_box, fps, speed_threshold=25):
    x1, y1, x2, y2, vid = vehicle_box
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    if vid not in speed_memory:
        speed_memory[vid] = (cx, cy)
        return False, 0.0

    px, py = speed_memory[vid]
    dist = math.hypot(cx - px, cy - py)

    speed = dist * fps  # pixels/sec
    speed_memory[vid] = (cx, cy)

    return speed > speed_threshold, speed

# ================= RED LIGHT JUMP =================
def detect_red_light_jump(vehicle_box, stop_line_y, light_is_red):
    x1, y1, x2, y2, vid = vehicle_box

    if not light_is_red:
        return False

    if y2 > stop_line_y and vid not in red_jump_memory:
        red_jump_memory.add(vid)
        return True

    return False
