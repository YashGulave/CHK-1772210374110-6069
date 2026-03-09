import numpy as np

class Sort:

    def __init__(self, max_age=20, min_hits=3, iou_threshold=0.3):
        self.id_count = 0

    def update(self, detections):

        results = []

        for det in detections:

            x1, y1, x2, y2, conf = det

            self.id_count += 1

            results.append([x1, y1, x2, y2, self.id_count])

        return np.array(results)