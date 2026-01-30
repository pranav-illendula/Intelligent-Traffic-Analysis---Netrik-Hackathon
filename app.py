import streamlit as st
import cv2
import tempfile
import pandas as pd

from v4_auto import detect_and_track
from queue_utils import get_queue_metrics
from violations import (
    detect_red_light_jump,
    detect_rash_driving,
    reset_violation_memory
)

# ================= STREAMLIT SETUP =================
st.set_page_config(page_title="Smart Traffic System", layout="wide")
st.title("🚦 Smart Traffic Monitoring & Violation Detection")

uploaded_video = st.file_uploader("Upload Traffic Video", type=["mp4", "avi"])

# ================= SIGNAL INFERENCE =================
def infer_signal_state(tracked_boxes, speed_map, stop_line_y):
    speeds = [
        speed_map.get(vid, 0)
        for (x1, y1, x2, y2, vid) in tracked_boxes
        if abs(y2 - stop_line_y) < 40
    ]

    if not speeds:
        return True  # default RED

    return sum(speeds) / len(speeds) < 3  # RED if vehicles stopped

# ================= MAIN =================
if uploaded_video:
    reset_violation_memory()

    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(uploaded_video.read())

    cap = cv2.VideoCapture(temp.name)
    fps = cap.get(cv2.CAP_PROP_FPS)

    video_col, info_col = st.columns([3, 1])
    frame_box = video_col.empty()

    qlen_m = info_col.metric("Queue Length", 0)
    qden_m = info_col.metric("Queue Density", 0.0)
    viol_m = info_col.metric("Violations", 0)

    csv_rows = []
    frame_no = 0
    speed_map = {}

    # ---- Output video writer (REPLAY ENABLED) ----
    ret, sample_frame = cap.read()
    h, w, _ = sample_frame.shape
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1
        STOP_LINE_Y = int(0.85 * h)

        tracked = detect_and_track(frame)

        # ---- SPEED CALCULATION ----
        for box in tracked:
            rash, speed = detect_rash_driving(box, fps)
            speed_map[box[4]] = speed

        light_is_red = infer_signal_state(tracked, speed_map, STOP_LINE_Y)

        q_len, q_den = get_queue_metrics(
            tracked, speed_map, STOP_LINE_Y, w
        )

        violations_count = 0

        for box in tracked:
            x1, y1, x2, y2, vid = box

            rash, speed = detect_rash_driving(box, fps)
            red = detect_red_light_jump(box, STOP_LINE_Y, light_is_red)

            color = (0, 255, 0)
            label = f"ID {vid}"

            if rash or red:
                color = (0, 0, 255)
                violations_count += 1
                label += " | "
                if red:
                    label += "RED JUMP "
                if rash:
                    label += "RASH "

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

            csv_rows.append({
                "frame": frame_no,
                "vehicle_id": vid,
                "speed(px/sec)": round(speed, 2),
                "red_light_jump": red,
                "rash_driving": rash,
                "queue_length": q_len,
                "queue_density": round(q_den, 3),
                "signal_state": "RED" if light_is_red else "GREEN"
            })

        # ---- DRAW STOP LINE ----
        line_color = (0, 0, 255) if light_is_red else (0, 255, 0)
        cv2.line(frame, (0, STOP_LINE_Y), (w, STOP_LINE_Y), line_color, 3)

        # ---- METRICS ----
        qlen_m.metric("Queue Length", q_len)
        qden_m.metric("Queue Density", round(q_den, 2))
        viol_m.metric("Violations", violations_count)

        frame_box.image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            use_container_width=True
        )

        out.write(frame)

    cap.release()
    out.release()

    df = pd.DataFrame(csv_rows)

    st.success("✅ Processing completed")

    st.video(out_path)

    st.download_button(
        "⬇️ Download Traffic CSV",
        df.to_csv(index=False).encode("utf-8"),
        "traffic_analysis.csv",
        "text/csv"
    )
