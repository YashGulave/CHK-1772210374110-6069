import time
import numpy as np
from ultralytics import YOLO
from violations.zebra_crossing import check_zebra_violation
import cv2
import cvzone
import math
import serial
from sort import Sort

# ----------------------------
# Arduino (optional)
# ----------------------------
# arduino = serial.Serial('COM3', 9600, timeout=1)

# ----------------------------
# Camera
# ----------------------------
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# ----------------------------
# YOLO Model
# ----------------------------
model = YOLO("Yolo-Weights/yolov8n.pt")

# ----------------------------
# SORT Tracker
# ----------------------------
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

active_ids = {}
total_count = 0
violation_count = 0
TTL = 90

# ----------------------------
# Classes
# ----------------------------
classNames = [
    "person", "bicycle", "car", "motorbike",
    "aeroplane", "bus", "train", "truck", "boat"
]

# ----------------------------
# Arduino Signal Function
# ----------------------------
def send_to_arduino(count):

    signal_time = min(count * 4, 60)

    print(f"Signal duration: {signal_time} seconds")

    # arduino.write(f"{signal_time}\n".encode())

    return signal_time


# ----------------------------
# Main Processing
# ----------------------------
def process_camera():

    global total_count, violation_count

    signal_active = False
    signal_start_time = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        # ----------------------------
        # YOLO Detection
        # ----------------------------
        results = model(frame, stream=True)

        detections = np.empty((0,5))

        for r in results:

            boxes = r.boxes

            for box in boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                conf = math.ceil((box.conf[0] * 100)) / 100

                cls = int(box.cls[0])

                if cls >= len(classNames):
                    continue

                current_class = classNames[cls]

                if current_class in {"car","truck","bus","motorbike"} and conf > 0.3:

                    detections = np.vstack((detections,[x1,y1,x2,y2,conf]))

                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

                    cvzone.putTextRect(
                        frame,
                        f"{current_class} {conf}",
                        (x1,y1-10),
                        scale=1,
                        thickness=2
                    )

        # ----------------------------
        # Tracking
        # ----------------------------
        results_tracker = tracker.update(detections)

        current_ids = set()

        for result in results_tracker:

            x1,y1,x2,y2,obj_id = map(int,result)

            current_ids.add(obj_id)

            cx = (x1+x2)//2
            cy = (y1+y2)//2

            # ----------------------------
            # Zebra Crossing Violation
            # ----------------------------
            if check_zebra_violation(cy):

                violation_count += 1

                cv2.putText(
                    frame,
                    "Zebra Crossing Violation",
                    (x1,y1-40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,0,255),
                    2
                )

                print(f"Vehicle {obj_id} crossed zebra line")


            # ----------------------------
            # Counting Vehicles
            # ----------------------------
            if obj_id not in active_ids:

                total_count += 1
                active_ids[obj_id] = 0

            else:

                active_ids[obj_id] = 0


        # ----------------------------
        # Remove inactive IDs
        # ----------------------------
        inactive_ids = [
            id_ for id_,frames in active_ids.items()
            if id_ not in current_ids and frames > TTL
        ]

        for id_ in inactive_ids:
            del active_ids[id_]


        for id_ in active_ids:
            if id_ not in current_ids:
                active_ids[id_] += 1


        # ----------------------------
        # Draw Zebra Crossing
        # ----------------------------
        cv2.line(frame,(0,500),(1280,500),(255,255,255),3)

        cv2.putText(
            frame,
            "Zebra Crossing",
            (10,480),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,255),
            2
        )


        # ----------------------------
        # Display Info
        # ----------------------------
        cv2.putText(
            frame,
            f"Traffic Count: {total_count}",
            (25,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2
        )

        cv2.putText(
            frame,
            f"Violations: {violation_count}",
            (25,100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2
        )


        cv2.imshow("Traffic Camera",frame)


        # ----------------------------
        # Signal Logic
        # ----------------------------
        if not signal_active and total_count > 0:

            signal_duration = send_to_arduino(total_count)

            total_count = 0

            active_ids.clear()

            signal_active = True

            signal_start_time = time.time()


        if signal_active and (time.time()-signal_start_time >= signal_duration):

            print("Signal cycle finished")

            signal_active = False


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# ----------------------------
# Run
# ----------------------------
process_camera()

cap.release()

cv2.destroyAllWindows()

# arduino.close()