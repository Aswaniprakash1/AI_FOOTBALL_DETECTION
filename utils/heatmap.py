






import cv2
import numpy as np
from pathlib import Path


class FootballHeatmap:
    """
    Football tactical heatmap generator.

    Features:
    - Accumulated player positions
    - Football pitch background
    - Red/yellow high-activity areas
    - Green/blue low-activity areas
    - Static PNG heatmap generation
    - Frame-by-frame tactical heatmap rendering
    """

    def __init__(self, pitch_width=640, pitch_height=360):

        self.pitch_width = pitch_width
        self.pitch_height = pitch_height

        # Accumulated player activity
        self.heatmap = np.zeros(
            (pitch_height, pitch_width),
            dtype=np.float32
        )

        self.position_count = 0

        # Pitch boundaries
        self.margin = 25

    # --------------------------------------------------
    # ADD PLAYER POSITION
    # --------------------------------------------------

    def add_position(self, x, y, frame_width, frame_height):
        """
        Convert player camera coordinates into pitch coordinates
        and accumulate player activity.
        """

        if frame_width <= 0 or frame_height <= 0:
            return

        margin = self.margin

        usable_width = self.pitch_width - 2 * margin
        usable_height = self.pitch_height - 2 * margin

        # Normalize player position
        normalized_x = np.clip(x / frame_width, 0, 1)
        normalized_y = np.clip(y / frame_height, 0, 1)

        # Map to tactical pitch
        pitch_x = int(
            margin + normalized_x * usable_width
        )

        pitch_y = int(
            margin + normalized_y * usable_height
        )

        pitch_x = np.clip(
            pitch_x,
            margin,
            self.pitch_width - margin
        )

        pitch_y = np.clip(
            pitch_y,
            margin,
            self.pitch_height - margin
        )

        # Add activity at player position
        cv2.circle(
            self.heatmap,
            (int(pitch_x), int(pitch_y)),
            radius=10,
            color=1.0,
            thickness=-1
        )

        self.position_count += 1

    # --------------------------------------------------
    # DRAW FOOTBALL PITCH
    # --------------------------------------------------

    def _draw_pitch(self):
        """
        Create green football pitch with white markings.
        """

        pitch = np.zeros(
            (
                self.pitch_height,
                self.pitch_width,
                3
            ),
            dtype=np.uint8
        )

        # Green grass
        pitch[:] = (35, 105, 35)

        white = (255, 255, 255)

        margin = self.margin

        left = margin
        top = margin

        right = self.pitch_width - margin
        bottom = self.pitch_height - margin

        center_x = self.pitch_width // 2
        center_y = self.pitch_height // 2

        # Outer pitch boundary
        cv2.rectangle(
            pitch,
            (left, top),
            (right, bottom),
            white,
            2
        )

        # Halfway line
        cv2.line(
            pitch,
            (center_x, top),
            (center_x, bottom),
            white,
            2
        )

        # Center circle
        cv2.circle(
            pitch,
            (center_x, center_y),
            55,
            white,
            2
        )

        # Center spot
        cv2.circle(
            pitch,
            (center_x, center_y),
            3,
            white,
            -1
        )

        # Left penalty area
        cv2.rectangle(
            pitch,
            (left, center_y - 85),
            (left + 90, center_y + 85),
            white,
            2
        )

        # Right penalty area
        cv2.rectangle(
            pitch,
            (right - 90, center_y - 85),
            (right, center_y + 85),
            white,
            2
        )

        # Left goal area
        cv2.rectangle(
            pitch,
            (left, center_y - 40),
            (left + 35, center_y + 40),
            white,
            2
        )

        # Right goal area
        cv2.rectangle(
            pitch,
            (right - 35, center_y - 40),
            (right, center_y + 40),
            white,
            2
        )

        # Penalty spots
        cv2.circle(
            pitch,
            (left + 65, center_y),
            3,
            white,
            -1
        )

        cv2.circle(
            pitch,
            (right - 65, center_y),
            3,
            white,
            -1
        )

        return pitch

    # --------------------------------------------------
    # RENDER TACTICAL HEATMAP FRAME
    # --------------------------------------------------

    def render_frame(self):
        """
        Generate one tactical heatmap frame.

        This method can be called repeatedly during video
        processing to create an accumulating heatmap video.
        """

        pitch = self._draw_pitch()

        # No activity detected
        if self.position_count == 0:
            return pitch

        # Smooth accumulated activity
        blurred = cv2.GaussianBlur(
            self.heatmap,
            (0, 0),
            sigmaX=18,
            sigmaY=18
        )

        maximum = float(blurred.max())

        if maximum <= 0:
            return pitch

        # Normalize activity
        normalized = blurred / maximum

        normalized = np.clip(
            normalized,
            0,
            1
        )

        # Apply heatmap color
        # Blue -> Green -> Yellow -> Red
        colored_heat = cv2.applyColorMap(
            (normalized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )

        # Heatmap transparency
        alpha = (
            normalized * 0.80
        )[..., None]

        # Blend heatmap with pitch
        pitch_float = pitch.astype(np.float32)

        heat_float = colored_heat.astype(np.float32)

        output = (
            pitch_float * (1 - alpha)
            + heat_float * alpha
        )

        output = np.clip(
            output,
            0,
            255
        ).astype(np.uint8)

        # Draw pitch markings above the heatmap
        markings = self._draw_pitch()

        margin = self.margin

        # Pitch border
        cv2.rectangle(
            output,
            (margin, margin),
            (
                self.pitch_width - margin,
                self.pitch_height - margin
            ),
            (255, 255, 255),
            2
        )

        # Center line
        cv2.line(
            output,
            (
                self.pitch_width // 2,
                margin
            ),
            (
                self.pitch_width // 2,
                self.pitch_height - margin
            ),
            (255, 255, 255),
            2
        )

        # Center circle
        cv2.circle(
            output,
            (
                self.pitch_width // 2,
                self.pitch_height // 2
            ),
            55,
            (255, 255, 255),
            2
        )

        # Activity information
        cv2.putText(
            output,
            f"PLAYER POSITIONS: {self.position_count}",
            (35, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        return output

    # --------------------------------------------------
    # GENERATE FINAL STATIC HEATMAP
    # --------------------------------------------------

    def generate_heatmap(self, output_path):
        """
        Save final accumulated tactical heatmap as PNG.
        """

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        frame = self.render_frame()

        success = cv2.imwrite(
            str(output_path),
            frame
        )

        if not success:
            raise RuntimeError(
                f"Heatmap image could not be saved: {output_path}"
            )

        return str(output_path)

    # --------------------------------------------------
    # RESET HEATMAP
    # --------------------------------------------------

    def reset(self):
        """
        Clear all accumulated player positions.
        """

        self.heatmap.fill(0)

        self.position_count = 0

    # --------------------------------------------------
    # GET POSITION COUNT
    # --------------------------------------------------

    def get_position_count(self):
        """
        Return number of accumulated player positions.
        """

        return self.position_count
