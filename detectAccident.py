import cv2
import numpy as np
import time
from ultralytics import YOLO

# Load model with tracking enabled
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("theVedioAccident.mp4")

# Store history per track ID
tracks = {}

COLLISION_DISTANCE = 70  # Moderate proximity
SPEED_THRESHOLD = 6      # Moderate speed threshold
ACCELERATION_THRESHOLD = 3  # Moderate acceleration threshold
IOU_THRESHOLD = 0.2      # Moderate IoU threshold
COOLDOWN = 60

last_capture_time = 0
frame_id = 0


def get_center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])


def distance(a, b):
    return np.linalg.norm(a - b)


def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    xi1 = max(x1, x3)
    yi1 = max(y1, y3)
    xi2 = min(x2, x4)
    yi2 = min(y2, y4)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)

    union = box1_area + box2_area - inter_area

    return inter_area / union if union > 0 else 0


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    # 🔥 Tracking (IMPORTANT FIX)
    results = model.track(frame, persist=True, verbose=False)[0]

    accident_detected = False
    current_data = {}

    if results.boxes.id is not None:

        for box, track_id, cls in zip(
            results.boxes.xyxy,
            results.boxes.id,
            results.boxes.cls
        ):
            track_id = int(track_id)
            cls = int(cls)

            if cls != 2:  # car only
                continue

            x1, y1, x2, y2 = map(int, box)
            center = get_center((x1, y1, x2, y2))

            # store history
            if track_id not in tracks:
                tracks[track_id] = []

            tracks[track_id].append(center)

            # keep last 5 frames only
            if len(tracks[track_id]) > 5:
                tracks[track_id].pop(0)

            current_data[track_id] = (center, (x1, y1, x2, y2))

            # draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 💥 Accident detection
    ids = list(current_data.keys())

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):

            id1, id2 = ids[i], ids[j]

            c1, b1 = current_data[id1]
            c2, b2 = current_data[id2]

            dist = distance(c1, c2)
            iou = compute_iou(b1, b2)

            # need history
            if len(tracks[id1]) < 2 or len(tracks[id2]) < 2:
                continue

            # velocity
            v1 = distance(tracks[id1][-1], tracks[id1][-2])
            v2 = distance(tracks[id2][-1], tracks[id2][-2])

            # acceleration (shock detection)
            a1 = abs(v1 - distance(tracks[id1][-2], tracks[id1][-3])) if len(tracks[id1]) >= 3 else 0
            a2 = abs(v2 - distance(tracks[id2][-2], tracks[id2][-3])) if len(tracks[id2]) >= 3 else 0

            # 🔥 BALANCED accident logic - require both vehicles slow AND significant collision evidence
            if (
                dist < COLLISION_DISTANCE and
                v1 < SPEED_THRESHOLD and v2 < SPEED_THRESHOLD and  # BOTH vehicles must be slow
                (abs(a1) > ACCELERATION_THRESHOLD or abs(a2) > ACCELERATION_THRESHOLD) and  # acceleration change
                iou > IOU_THRESHOLD  # AND overlap
            ):
                accident_detected = True
                print(f"🚨 ACCIDENT DETECTED: ID{id1} & ID{id2} - Dist:{dist:.1f}, IoU:{iou:.2f}, V1:{v1:.1f}, V2:{v2:.1f}, A1:{a1:.1f}, A2:{a2:.1f}")
            # Removed debug prints for cleaner output

    # 📸 Screenshot (cooldown)
    current_time = time.time()

    if accident_detected and (current_time - last_capture_time > COOLDOWN):
        filename = f"images/accident_{frame_id}.jpg"
        cv2.imwrite(filename, frame)
        print(f"🚨 ACCIDENT SAVED: {filename}")
        last_capture_time = current_time

    if accident_detected:
        cv2.putText(frame, "ACCIDENT DETECTED!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Accident Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()