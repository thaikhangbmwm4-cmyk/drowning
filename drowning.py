import json
import signal
import sys
import threading
import requests
import cv2
import numpy as np
from picamera2 import Picamera2
import time
from ultralytics import YOLO
from collections import deque
import os
from analysis.temporal_collector import (
    create_temporal_state,
    record_observation,
    update_tracking_state,
    DetectionIntervalTracker,
)

SHOW = False      # True nếu cắm màn hình, False khi chạy từ bot (không có cửa sổ)
running = True
os.makedirs("alerts", exist_ok=True)
video_buffer = deque()
BUFFER_SECONDS = 20
TOKEN = "8902284441:AAFGeIWRcXY9bvzdRZOwwJYFn_UBgN2G0js"
CHAT_ID = -5474359022
MOVE_THRESHOLD = 1
MAX_MATCH_DISTANCE = 60
last = time.monotonic()
next_id = 1
frame_count = 0
people = {}
detection_clock = DetectionIntervalTracker()
recently_lost = deque(maxlen=20)

def stop_handler(*args):      # bot gửi 'stop' -> vòng lặp tự thoát
    global running
    running = False
signal.signal(signal.SIGTERM, stop_handler)

def send_photo_async(file_path, caption="ALERT"):
    def task():
        try:
            url = (f"https://api.telegram.org/bot{TOKEN}/sendPhoto")
            with open(file_path, "rb") as f:
                files = {"photo": f}
                data = {"chat_id": CHAT_ID, "caption": caption}
                requests.post(url, files=files, data=data, timeout=60)
        except Exception as e:
            print(e)

    threading.Thread(target=task, daemon=True).start()

def send_video_async(file_path, caption="ALERT"):
    def task():
        try:
            url = f"https://api.telegram.org/bot{TOKEN}/sendVideo"
            with open(file_path, "rb") as f:
                files = {"video": f}
                data = {"chat_id": CHAT_ID, "caption": caption}
                requests.post(url, files=files, data=data, timeout=60)
        except Exception as e:
            print("Telegram send error:", e)

    threading.Thread(target=task, daemon=True).start()

def get_event_video(buffer, event_time, pre=5, post=5):
    frames = []
    for t, f in buffer:
        if event_time - pre <= t <= event_time + post:
            frames.append(f)
    return frames

def save_alert_video(frames):
    if len(frames) == 0:
        return None
    filename = f"alerts/alert_{int(time.time())}.mp4"
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"mp4v"), 4, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    return filename

def in_water(pt):
    return cv2.pointPolygonTest(water_zone, pt, False) >= 0

def movement_score(history, box):
    if len(history) < 5:
        return 999
    recent = list(history)[-5:]
    total = 0
    for i in range(1, len(recent)):
        step = distance(
            recent[i - 1],
            recent[i]
        )
        if step > 2:
            total += step
    x1, y1, x2, y2 = box
    box_h = max(1, y2 - y1)
    return total / box_h

def moving_toward_edge(history, roi, box, near_ratio=1.5, min_move_ratio=0.08):
    if len(history) < 5:
        return False
    old_pt = history[-5]
    new_pt = history[-1]
    x1, y1, x2, y2 = box
    box_h = max(1, y2 - y1)
    if cv2.pointPolygonTest(
            roi,
            new_pt,
            False
    ) < 0:
        return False
    old_dist = abs(
        cv2.pointPolygonTest(
            roi,
            old_pt,
            True
        )
    )
    new_dist = abs(
        cv2.pointPolygonTest(
            roi,
            new_pt,
            True
        )
    )
    move = distance(
        old_pt,
        new_pt
    )
    edge_ratio = new_dist / box_h
    move_ratio = move / box_h
    if move_ratio < min_move_ratio:
        return False
    if edge_ratio > near_ratio:
        return False
    return new_dist < old_dist

def distance(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return (dx * dx + dy * dy) ** 0.5

def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0

# =========================
#  LIGHT MODEL (IMPORTANT)
# =========================

model = YOLO("drowning_ncnn_model")
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

# =========================
#          ROI
# =========================

if not os.path.exists("roi.json"):
    print("Chưa có roi.json")
    picam2.stop()
    sys.exit(1)

with open("roi.json") as f:
    roi = json.load(f)

sx = 640 / roi["width"]      # quy đổi nếu kích thước ảnh khác nhau
sy = 480 / roi["height"]
water_zone = np.array(
    [[int(x * sx), int(y * sy)] for x, y in roi["points"]],
    np.int32
)

# =========================
# MAIN LOOP (OPTIMIZED FPS)
# =========================

while running:
    frame_count += 1

    if frame_count % 2 != 0:
        continue

    detections = []
    filtered = []

    frame = picam2.capture_array()

    if frame is None:
        print("Camera fail frame")
        continue

    # Lấy thời gian sau khi có frame hợp lệ
    frame_time = time.monotonic()
    wall_time = time.time()
    now = frame_time

    dt = frame_time - last
    last = frame_time
    fps = 1 / dt if dt > 0 else 0

    if frame_count % 100 == 0:
        print(f"FPS: {fps:.2f}")

    detection_clock.update(frame_time)
    missing_threshold = detection_clock.get_missing_threshold()
    video_buffer.append((frame_time, frame.copy()))
    while (
            video_buffer
            and frame_time, - video_buffer[0][0] > BUFFER_SECONDS
    ):
        video_buffer.popleft()

    cv2.polylines(
        frame,
        [water_zone],
        True,
        (0, 255, 255),
        2
    )
    results = model(
        frame,
        imgsz=320,
        classes=[0],
        conf=0.3,
        verbose=False
    )
    if results is None:
        continue
    used_ids = set()
    recently_lost = deque(
        [x for x in recently_lost if now - x["time"] <= 5],
        maxlen=20
    )

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            detections.append({
                "box": (x1, y1, x2, y2),
                "conf": float(box.conf[0])
            })

    for det in sorted(
            detections,
            key=lambda x: x["conf"],
            reverse=True
    ):
        duplicate = False
        for old in filtered:
            if box_iou(
                    det["box"],
                    old["box"]
            ) > 0.5:
                duplicate = True
                break

        if not duplicate:
            filtered.append(det)
    for det in filtered:
        x1, y1, x2, y2 = det["box"]
        center = (
            int((x1 + x2) / 2),
            int((y1 + y2) / 2)
        )

        # =======================
        #    BYTE TRACK LIGHT
        # =======================

        best_id = None
        best_score = 999999
        for old_pid, old_person in people.items():
            if old_person["last_pos"] is None:
                continue
            lost_time = now - old_person["last_seen"]
            if lost_time > 2:
                continue
            if old_pid in used_ids:
                continue
            dist = distance(
                center,
                old_person["last_pos"]
            )
            iou = 0
            if old_person["last_box"] is not None:
                iou = box_iou(
                    (x1, y1, x2, y2),
                    old_person["last_box"]
                )
            score = dist - iou * 50
            if score < best_score:
                best_score = score
                best_id = old_pid
        if best_id is not None and best_score < MAX_MATCH_DISTANCE:
            pid = best_id
            used_ids.add(pid)
        else:
            pid = next_id
            next_id += 1

        # =======================
        #         REID
        # =======================

        if pid not in people:
            matched_old_pid = None
            for lost in list(recently_lost):
                dist = distance(center, lost["pos"])
                if dist < MAX_MATCH_DISTANCE and now - lost["time"] < 5:
                    matched_old_pid = lost["old_pid"]
                    break
            while len(recently_lost) > 0 and now - recently_lost[0]["time"] > 5:
                recently_lost.popleft()
            if matched_old_pid is not None and matched_old_pid in people:
                pid = matched_old_pid

                people[pid]["lost_recorded"] = False
                people[pid]["lost_start_time"] = None
                people[pid]["video_sent"] = False

                for item in list(recently_lost):
                    if item["old_pid"] == matched_old_pid:
                        recently_lost.remove(item)
                        break
            else:
                people[pid] = {
                    "last_seen": frame_time,
                    "last_pos": None,
                    "track_history": deque(maxlen=20),
                    "last_water_time": None,
                    "video_sent": False,
                    "lost_start_time": None,
                    "in_water": False,
                    "lost_recorded": False,
                    "last_box": None,
                    **create_temporal_state()
                }
        p = people[pid]
        p["last_seen"] = frame_time
        p["lost_recorded"] = False

        # =======================
        #    UPDATE TRACK
        # =======================

        inside = in_water(center)
        if inside:
            p["last_water_time"] = frame_time
        p["last_pos"] = center
        p["last_box"] = (x1, y1, x2, y2)
        p["track_history"].append(center)
        record_observation(
            person=p,
            now=frame_time,
            wall_time=wall_time,
            center=center,
            box=(x1, y1, x2, y2),
            confidence=det["conf"],
            inside_roi=inside,
        )
        # =======================
        #   DRAW DETECTION
        # =======================

        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2
        )
        cv2.putText(
            frame,
            f"ID {pid}",
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
    for pid, p in people.items():
        if p["last_pos"] is None:
            continue

        # =======================
        #   PERSON ANALYSIS
        # =======================

        lost_time = update_tracking_state(p, frame_time, missing_threshold)
        if lost_time is None:
            continue
        if p["tracking_state"] == "missing" and frame_count % 20 == 0:
            print(f"ID {pid} MISSING {lost_time:.1f}s | "
                  f"lost_pos={p['lost_pos']} | trong_ROI={p['lost_inside_roi']}")
        x, y = p["last_pos"]
        moving_edge = moving_toward_edge(p["track_history"], water_zone, p["last_box"])
        move_score = movement_score(p["track_history"], p["last_box"])
        recent_in_water = (
            p["last_water_time"] is not None
            and now - p["last_water_time"] < 15
        )

        # =======================
        #   LOST DETECTION
        # =======================

        if lost_time > 2 and not p["lost_recorded"]:
            p["lost_start_time"] = p["last_seen"]
            p["lost_recorded"] = True
            recently_lost.append({
                "old_pid": pid,
                "pos": p["last_pos"],
                "time": now
            })

        # =======================
        #  DROWNING DETECTION
        # =======================

        if (
                recent_in_water
                and lost_time > 8
                and not moving_edge
                and move_score < MOVE_THRESHOLD
        ):

            # =======================
            #   CREATE EVIDENCE
            # =======================

            snapshot = frame.copy()
            cv2.circle(
                snapshot,
                p["last_pos"],
                12,
                (0, 0, 255),
                -1
            )
            cv2.putText(
                snapshot,
                "LAST SEEN",
                (p["last_pos"][0] + 10,
                 p["last_pos"][1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2)
            image_file = f"alerts/alert_{int(time.time())}.jpg"
            if not p["video_sent"]:
                if p["lost_start_time"] is None:
                    continue
                frames = get_event_video(
                    video_buffer,
                    p["lost_start_time"],
                    pre=5,
                    post=5
                )
                if len(frames) > 0:
                    cv2.imwrite(image_file, snapshot)
                    alert_video = save_alert_video(frames)
                    if alert_video is None:
                        continue

                    # =======================
                    #   ALERT TELEGRAM
                    # =======================

                    send_photo_async(
                        image_file,
                        caption=f"VỊ TRÍ CUỐI CÙNG ID {pid}"
                    )
                    send_video_async(
                        alert_video,
                        caption=f"NGHI NGỜ ĐUỐI NƯỚC ID {pid}"
                    )
                    p["video_sent"] = True
                    print(f"NGHI NGỜ ĐUỐI NƯỚC ID {pid}")

    # =======================
    #    TRACK CLEANUP
    # =======================

    remove_ids = []
    for pid, p in people.items():
        lost_time = now - p["last_seen"]
        if lost_time > 15:
            remove_ids.append(pid)
    for pid in remove_ids:
        del people[pid]

    # =======================
    #      DISPLAY
    # =======================

    cv2.imshow("Raspberry Pi Drowning AI", frame)
    if cv2.waitKey(1) == 27:
        break
picam2.stop()
cv2.destroyAllWindows()
"""
    if SHOW:
        cv2.imshow("Raspberry Pi Drowning AI", frame)
        if cv2.waitKey(1) == 27:
            break

picam2.stop()
picam2.close()
cv2.destroyAllWindows()
"""
