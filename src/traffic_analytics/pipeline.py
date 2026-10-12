from ultralytics import YOLO
import cv2

# ================= CONFIG =================
COCO_MODEL_PATH = "yolov8s.pt"
AUTO_MODEL_PATH = "yolov8n.pt"

CONF_MARGIN = 0.15
IOU_MATCH = 0.5
COCO_VEHICLES = {"car", "bus", "truck", "motorcycle"}

coco_model = YOLO(COCO_MODEL_PATH)
auto_model = YOLO(AUTO_MODEL_PATH)

# ================= IOU =================
def iou(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (areaA + areaB - inter + 1e-6)


# ================= MAIN FUNCTION =================
def detect_and_track(frame):
    """
    Extracted from notebook8fb321dd9d (1).ipynb
    Returns: [(x1, y1, x2, y2, track_id)]
    """

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ===== COCO DETECTION + TRACKING =====
    coco_res = coco_model.track(
        frame_rgb, persist=True, conf=0.25, verbose=False
    )[0]

    vehicle_boxes = []

    if coco_res.boxes is not None:
        for b in coco_res.boxes:
            label = coco_res.names[int(b.cls)]
            conf = float(b.conf)
            track_id = int(b.id) if b.id is not None else None

            if label in COCO_VEHICLES:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                vehicle_boxes.append((x1, y1, x2, y2, label, conf, track_id))

    # ===== CROP FOR AUTO MODEL =====
    crops = []
    crop_coords = []

    for (x1, y1, x2, y2, _, _, _) in vehicle_boxes:
        crop = frame_rgb[y1:y2, x1:x2]
        if crop.size > 0:
            crops.append(crop)
            crop_coords.append((x1, y1))

    auto_results = []
    if len(crops) > 0:
        auto_results = auto_model.predict(crops, conf=0.25, verbose=False)

    tracked_autos = []
    for i, res in enumerate(auto_results):
        if res.boxes is None:
            continue
        xoff, yoff = crop_coords[i]
        for b in res.boxes:
            ax1, ay1, ax2, ay2 = b.xyxy[0]
            tracked_autos.append((
                int(xoff + ax1),
                int(yoff + ay1),
                int(xoff + ax2),
                int(yoff + ay2),
                float(b.conf)
            ))

    # ===== CONFIDENCE FUSION =====
    final_boxes = [
        [x1, y1, x2, y2, label, conf, tid]
        for (x1, y1, x2, y2, label, conf, tid) in vehicle_boxes
    ]

    for (ax1, ay1, ax2, ay2, aconf) in tracked_autos:
        best_iou = 0
        best_idx = -1

        for i, (cx1, cy1, cx2, cy2, _, cconf, _) in enumerate(final_boxes):
            overlap = iou((ax1, ay1, ax2, ay2), (cx1, cy1, cx2, cy2))
            if overlap > best_iou:
                best_iou = overlap
                best_idx = i

        if best_iou >= IOU_MATCH:
            if aconf >= final_boxes[best_idx][5] + CONF_MARGIN:
                final_boxes[best_idx][0:4] = [ax1, ay1, ax2, ay2]
                final_boxes[best_idx][5] = aconf

    # ===== RETURN TRACKED BOXES =====
    tracked_boxes = []
    for (x1, y1, x2, y2, _, _, tid) in final_boxes:
        if tid is not None:
            tracked_boxes.append((x1, y1, x2, y2, tid))

    return tracked_boxes
