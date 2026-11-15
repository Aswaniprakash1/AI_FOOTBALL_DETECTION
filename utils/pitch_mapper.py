
import cv2
import numpy as np

from config import PITCH_WIDTH, PITCH_HEIGHT


class PitchMapper:

    def __init__(self):

        self.pitch_width = PITCH_WIDTH
        self.pitch_height = PITCH_HEIGHT

        self.homography = None

        # Top-down pitch corner coordinates
        self.pitch_points = np.array([
            [0, 0],
            [self.pitch_width, 0],
            [self.pitch_width, self.pitch_height],
            [0, self.pitch_height]
        ], dtype=np.float32)

    def calibrate(self, image_points):

        image_points = np.array(
            image_points,
            dtype=np.float32
        )

        if image_points.shape != (4, 2):
            raise ValueError(
                "Provide exactly four pitch corner points."
            )

        self.homography = cv2.getPerspectiveTransform(
            image_points,
            self.pitch_points
        )

        return self.homography

    def map_position(self, position):

        if self.homography is None:
            return None

        point = np.array(
            [[position]],
            dtype=np.float32
        )

        mapped = cv2.perspectiveTransform(
            point,
            self.homography
        )

        x, y = mapped[0][0]

        return float(x), float(y)

    def draw_pitch(self, size=(700, 450)):

        width, height = size

        pitch = np.zeros(
            (height, width, 3),
            dtype=np.uint8
        )

        pitch[:] = (40, 120, 40)

        margin = 20

        cv2.rectangle(
            pitch,
            (margin, margin),
            (width - margin, height - margin),
            (255, 255, 255),
            2
        )

        center_x = width // 2

        cv2.line(
            pitch,
            (center_x, margin),
            (center_x, height - margin),
            (255, 255, 255),
            2
        )

        cv2.circle(
            pitch,
            (center_x, height // 2),
            50,
            (255, 255, 255),
            2
        )

        cv2.circle(
            pitch,
            (center_x, height // 2),
            4,
            (255, 255, 255),
            -1
        )

        return pitch
