

   


import cv2
from ultralytics import YOLO

from config import (
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    TRACKER_CONFIG,
    TEAM_A_COLOR,
    TEAM_B_COLOR,
    REFEREE_COLOR,
    BALL_COLOR
)

from utils.team_classifier import TeamClassifier


class FootballTracker:

    def __init__(self):

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}"
            )

        self.model = YOLO(str(MODEL_PATH))

        self.team_classifier = TeamClassifier()

        self.class_names = self.model.names

        print("Football YOLO model loaded successfully")

    # ========================================================
    # CLASS NAME
    # ========================================================

    def get_class_name(self, class_id):

        if isinstance(self.class_names, dict):

            return str(
                self.class_names.get(
                    class_id,
                    class_id
                )
            ).lower()

        return str(
            self.class_names[class_id]
        ).lower()

    # ========================================================
    # OBJECT CLASSIFICATION
    # ========================================================

    def classify_object(self, class_name):

        name = class_name.lower()

        if "referee" in name or "ref" in name:
            return "referee"

        if "goalkeeper" in name or "keeper" in name:
            return "goalkeeper"

        if "ball" in name or "football" in name:
            return "ball"

        if "player" in name:
            return "player"

        return "other"

    # ========================================================
    # PLAYER FALSE DETECTION FILTER
    # ========================================================

    def filter_false_players(self, detections):

        filtered_detections = []

        for detection in detections:

            if detection["object_type"] == "player":

                x1, y1, x2, y2 = detection["bbox"]

                box_width = x2 - x1
                box_height = y2 - y1

                box_area = (
                    box_width * box_height
                )

                # Remove extremely small detections
                if box_height < 25:
                    continue

                if box_width < 8:
                    continue

                if box_area < 350:
                    continue

                # Remove extremely wide player boxes
                aspect_ratio = (
                    box_width / box_height
                    if box_height > 0
                    else 0
                )

                if aspect_ratio > 1.8:
                    continue

            filtered_detections.append(
                detection
            )

        return filtered_detections

    # ========================================================
    # PLAYER TRACKING
    # ========================================================

    def track(self, frame, confidence=None):

        conf_value = (
            confidence
            if confidence is not None
            else CONFIDENCE_THRESHOLD
        )

        results = self.model.track(
            frame,
            persist=True,
            conf=conf_value,
            tracker=TRACKER_CONFIG,
            verbose=False
        )

        detections = []

        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes

        # ----------------------------------------------------
        # EXTRACT DETECTIONS
        # ----------------------------------------------------

        for box in boxes:

            xyxy = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )

            class_id = int(
                box.cls[0].item()
            )

            confidence_value = float(
                box.conf[0].item()
            )

            class_name = self.get_class_name(
                class_id
            )

            object_type = self.classify_object(
                class_name
            )

            track_id = None

            if box.id is not None:

                track_id = int(
                    box.id[0].item()
                )

            detections.append({

                "bbox": xyxy,

                "class_id": class_id,

                "class_name": class_name,

                "object_type": object_type,

                "confidence": confidence_value,

                "track_id": track_id

            })

        # ----------------------------------------------------
        # FILTER FALSE PLAYER DETECTIONS
        # ----------------------------------------------------

        detections = self.filter_false_players(
            detections
        )

        # ----------------------------------------------------
        # TEAM CLASSIFICATION
        # ----------------------------------------------------

        detections = (
            self.team_classifier.classify_teams(
                frame,
                detections
            )
        )

        return detections

    # ========================================================
    # DRAW DETECTIONS
    # ========================================================

    def draw_detections(
        self,
        frame,
        detections
    ):

        for detection in detections:

            x1, y1, x2, y2 = detection["bbox"]

            object_type = detection["object_type"]

            track_id = detection["track_id"]

            confidence = detection["confidence"]

            team = detection.get("team")

            # ------------------------------------------------
            # SELECT COLOR AND LABEL
            # ------------------------------------------------

            if object_type == "referee":

                color = REFEREE_COLOR
                label = "REFEREE"

            elif object_type == "ball":

                color = BALL_COLOR
                label = "BALL"

            elif object_type == "goalkeeper":

                color = TEAM_B_COLOR
                label = "GOALKEEPER"

            elif object_type == "player":

                if team == 0:

                    color = TEAM_A_COLOR
                    label = "TEAM A"

                elif team == 1:

                    color = TEAM_B_COLOR
                    label = "TEAM B"

                else:

                    color = TEAM_A_COLOR
                    label = "PLAYER"

            else:

                color = TEAM_A_COLOR
                label = object_type.upper()

            # ------------------------------------------------
            # TRACK ID
            # ------------------------------------------------

            if track_id is not None:

                label += f" ID:{track_id}"

            label += f" {confidence:.2f}"

            # ------------------------------------------------
            # DRAW BOUNDING BOX
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # ------------------------------------------------
            # DRAW LABEL
            # ------------------------------------------------

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        return frame
