
import cv2
import numpy as np


class TeamClassifier:

    def __init__(self):
        self.team_colors = {}
        self.player_teams = {}

        # Stable mapping from K-means clusters to team IDs
        self.cluster_to_team = {}

    # --------------------------------------------------
    # EXTRACT JERSEY COLOR
    # --------------------------------------------------

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

        jersey = player_crop[
            int(player_crop.shape[0] * 0.15):
            int(player_crop.shape[0] * 0.60),
            int(player_crop.shape[1] * 0.20):
            int(player_crop.shape[1] * 0.80)
        ]

        if jersey.size == 0:
            return None

        hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)

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

        return np.median(
            pixels,
            axis=0
        ).astype(np.float32)

    # --------------------------------------------------
    # CLASSIFY TEAMS
    # --------------------------------------------------

    def classify_teams(self, frame, detections):

        jersey_colors = []

        for detection in detections:

            if detection.get("object_type") != "player":
                continue

            track_id = detection.get("track_id")

            if track_id is None:
                continue

            color = self.extract_jersey_color(
                frame,
                detection["bbox"]
            )

            if color is not None:
                jersey_colors.append(
                    (track_id, color)
                )

        # Need at least two players to cluster
        if len(jersey_colors) >= 2:

            pixels = np.array(
                [item[1] for item in jersey_colors],
                dtype=np.float32
            )

            criteria = (
                cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER,
                20,
                1.0
            )

            _, labels, centers = cv2.kmeans(
                pixels,
                2,
                None,
                criteria,
                5,
                cv2.KMEANS_PP_CENTERS
            )

            # Maintain stable cluster-to-team mapping
            if not self.cluster_to_team:

                # Initial cluster labels become team IDs
                self.cluster_to_team = {
                    0: 0,
                    1: 1
                }

            else:

                # Match new clusters to previous jersey colors
                old_centers = self.team_colors

                if 0 in old_centers and 1 in old_centers:

                    direct_distance = (
                        np.linalg.norm(centers[0] - old_centers[0]) +
                        np.linalg.norm(centers[1] - old_centers[1])
                    )

                    swapped_distance = (
                        np.linalg.norm(centers[0] - old_centers[1]) +
                        np.linalg.norm(centers[1] - old_centers[0])
                    )

                    if swapped_distance < direct_distance:
                        self.cluster_to_team = {
                            0: 1,
                            1: 0
                        }
                    else:
                        self.cluster_to_team = {
                            0: 0,
                            1: 1
                        }

            new_team_colors = {}

            for cluster_id in (0, 1):

                team_id = self.cluster_to_team[cluster_id]

                new_team_colors[team_id] = centers[cluster_id]

            self.team_colors = new_team_colors

            for index, (track_id, _) in enumerate(jersey_colors):

                cluster_id = int(labels[index][0])

                team_id = self.cluster_to_team[cluster_id]

                self.player_teams[track_id] = team_id

        # --------------------------------------------------
        # APPLY TEAM LABELS TO DETECTIONS
        # --------------------------------------------------

        for detection in detections:

            if detection.get("object_type") != "player":
                continue

            track_id = detection.get("track_id")

            team_id = self.player_teams.get(track_id)

            if team_id == 0:
                detection["team"] = "Team A"

            elif team_id == 1:
                detection["team"] = "Team B"

            else:
                detection["team"] = None

        return detections

    # --------------------------------------------------
    # GET PLAYER TEAM
    # --------------------------------------------------

    def get_player_team(self, track_id):

        team_id = self.player_teams.get(track_id)

        if team_id == 0:
            return "Team A"

        if team_id == 1:
            return "Team B"

        return None



