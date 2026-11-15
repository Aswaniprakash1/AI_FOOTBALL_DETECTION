


import cv2
import numpy as np


class TeamClassifier:

    def __init__(self):
        self.team_colors = {}
        self.player_teams = {}

    def extract_jersey_color(self, frame, bbox):

        x1, y1, x2, y2 = map(int, bbox)

        height, width = frame.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        player_crop = frame[y1:y2, x1:x2]

        if player_crop.size == 0:
            return None

        # Approximate upper-body / jersey region
        jersey = player_crop[
            int(player_crop.shape[0] * 0.15):
            int(player_crop.shape[0] * 0.60),
            int(player_crop.shape[1] * 0.20):
            int(player_crop.shape[1] * 0.80)
        ]

        if jersey.size == 0:
            return None

        hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)

        # Ignore dark pixels and grass-like green pixels
        mask = (
            (hsv[:, :, 2] > 50) &
            ~(
                (hsv[:, :, 0] >= 35) &
                (hsv[:, :, 0] <= 85)
            )
        )

        pixels = hsv[mask]

        if len(pixels) == 0:
            return None

        # Median HSV jersey color
        color = np.median(pixels, axis=0)

        return color.astype(np.float32)

    def classify_teams(self, frame, detections):

        jersey_colors = []

        for detection in detections:

            if detection["object_type"] != "player":
                continue

            color = self.extract_jersey_color(
                frame,
                detection["bbox"]
            )

            if color is not None:
                jersey_colors.append(
                    (detection["track_id"], color)
                )

        if len(jersey_colors) < 2:
            return detections

        pixels = np.array(
            [item[1] for item in jersey_colors],
            dtype=np.float32
        )

        # Two-team color clustering
        criteria = (
            cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER,
            20,
        1.0
        )

        compactness, labels, centers = cv2.kmeans(
            pixels,
            2,
            None,
            criteria,
            5,
            cv2.KMEANS_PP_CENTERS
        )

        self.team_colors = {
            0: centers[0],
            1: centers[1]
        }

        for index, (track_id, color) in enumerate(jersey_colors):

            team = int(labels[index][0])

            if track_id is not None:
                self.player_teams[track_id] = team

        for detection in detections:

            track_id = detection["track_id"]

            if track_id in self.player_teams:

                detection["team"] = (
                    self.player_teams[track_id]
                )

            else:
                detection["team"] = None

        return detections
