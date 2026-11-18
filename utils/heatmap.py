
import cv2
import numpy as np
from pathlib import Path


class FootballHeatmap:

    def __init__(self, pitch_width=800, pitch_height=520):
        self.pitch_width = pitch_width
        self.pitch_height = pitch_height

        # Original video coordinate history for video overlays
        self.position_history = []
        self.last_frame_index = 0

        # Scaled tactical pitch positions
        self.combined_positions = []
        self.team_a_positions = []
        self.team_b_positions = []

        # Position counters
        self.combined_count = 0
        self.team_a_count = 0
        self.team_b_count = 0

    # --------------------------------------------------
    # TEAM NORMALIZATION
    # --------------------------------------------------

    def normalize_team(self, team):

        if team is None:
            return None

        value = str(team).strip().lower()

        if value in ["0", "a", "team a", "team_a"]:
            return 0

        if value in ["1", "b", "team b", "team_b"]:
            return 1

        return None

    # --------------------------------------------------
    # ADD PLAYER POSITION
    # --------------------------------------------------

    def add_position(
        self,
        x,
        y,
        frame_width,
        frame_height,
        team=None,
        frame_index=None
    ):

        if frame_width <= 0 or frame_height <= 0:
            return

        team_id = self.normalize_team(team)

        # Original video coordinates
        video_x = int(x)
        video_y = int(y)

        video_x = max(0, min(frame_width - 1, video_x))
        video_y = max(0, min(frame_height - 1, video_y))

        # Convert original video coordinates to tactical pitch coordinates
        pitch_x = int((video_x / frame_width) * self.pitch_width)
        pitch_y = int((video_y / frame_height) * self.pitch_height)

        pitch_x = max(0, min(self.pitch_width - 1, pitch_x))
        pitch_y = max(0, min(self.pitch_height - 1, pitch_y))

        if frame_index is None:
            frame_index = self.last_frame_index + 1

        self.last_frame_index = max(
            self.last_frame_index,
            frame_index
        )

        # Store original video coordinates for video heatmap overlays
        record = (
            frame_index,
            video_x,
            video_y,
            team_id
        )

        self.position_history.append(record)

        # Store tactical pitch coordinates
        self.combined_positions.append((pitch_x, pitch_y))
        self.combined_count += 1

        if team_id == 0:
            self.team_a_positions.append((pitch_x, pitch_y))
            self.team_a_count += 1

        elif team_id == 1:
            self.team_b_positions.append((pitch_x, pitch_y))
            self.team_b_count += 1

    # --------------------------------------------------
    # PITCH DRAWING
    # --------------------------------------------------

    def draw_pitch(self):

        pitch = np.zeros(
            (self.pitch_height, self.pitch_width, 3),
            dtype=np.uint8
        )

        pitch[:] = (40, 125, 40)

        white = (255, 255, 255)
        thickness = 2
        margin = 25

        # Outer boundary
        cv2.rectangle(
            pitch,
            (margin, margin),
            (self.pitch_width - margin, self.pitch_height - margin),
            white,
            thickness
        )

        center_x = self.pitch_width // 2
        center_y = self.pitch_height // 2

        # Halfway line
        cv2.line(
            pitch,
            (center_x, margin),
            (center_x, self.pitch_height - margin),
            white,
            thickness
        )

        # Center circle
        cv2.circle(
            pitch,
            (center_x, center_y),
            65,
            white,
            thickness
        )

        # Center spot
        cv2.circle(
            pitch,
            (center_x, center_y),
            3,
            white,
            -1
        )

        # Penalty boxes
        box_depth = 115
        box_height = 210

        cv2.rectangle(
            pitch,
            (margin, center_y - box_height // 2),
            (margin + box_depth, center_y + box_height // 2),
            white,
            thickness
        )

        cv2.rectangle(
            pitch,
            (self.pitch_width - margin - box_depth, center_y - box_height // 2),
            (self.pitch_width - margin, center_y + box_height // 2),
            white,
            thickness
        )

        # Goal areas
        goal_depth = 45
        goal_height = 100

        cv2.rectangle(
            pitch,
            (margin, center_y - goal_height // 2),
            (margin + goal_depth, center_y + goal_height // 2),
            white,
            thickness
        )

        cv2.rectangle(
            pitch,
            (self.pitch_width - margin - goal_depth, center_y - goal_height // 2),
            (self.pitch_width - margin, center_y + goal_height // 2),
            white,
            thickness
        )

        # Goals
        cv2.rectangle(
            pitch,
            (margin - 8, center_y - 35),
            (margin, center_y + 35),
            white,
            thickness
        )

        cv2.rectangle(
            pitch,
            (self.pitch_width - margin, center_y - 35),
            (self.pitch_width - margin + 8, center_y + 35),
            white,
            thickness
        )

        return pitch

    # --------------------------------------------------
    # PITCH HEATMAP
    # --------------------------------------------------

    def render_map(self, position_map):

        pitch = self.draw_pitch()

        if position_map is None or len(position_map) == 0:
            return pitch

        heat = np.zeros(
            (self.pitch_height, self.pitch_width),
            dtype=np.float32
        )

        for x, y in position_map:

            if 0 <= x < self.pitch_width and 0 <= y < self.pitch_height:

                cv2.circle(
                    heat,
                    (int(x), int(y)),
                    25,
                    1.0,
                    -1
                )

        heat = cv2.GaussianBlur(
            heat,
            (0, 0),
            sigmaX=25,
            sigmaY=25
        )

        if heat.max() > 0:

            heat = cv2.normalize(
                heat,
                None,
                0,
                255,
                cv2.NORM_MINMAX
            ).astype(np.uint8)

            colored = cv2.applyColorMap(
                heat,
                cv2.COLORMAP_JET
            )

            mask = heat > 10

            pitch[mask] = cv2.addWeighted(
                pitch[mask],
                0.35,
                colored[mask],
                0.65,
                0
            )

        # Draw pitch markings on top of the heatmap
        pitch = self.draw_pitch_over_heat(pitch)

        return pitch

    # --------------------------------------------------
    # DRAW PITCH MARKINGS OVER HEAT
    # --------------------------------------------------

    def draw_pitch_over_heat(self, image):

        white = (255, 255, 255)
        thickness = 2
        margin = 25

        center_x = self.pitch_width // 2
        center_y = self.pitch_height // 2

        # Outer boundary
        cv2.rectangle(
            image,
            (margin, margin),
            (self.pitch_width - margin, self.pitch_height - margin),
            white,
            thickness
        )

        # Halfway line
        cv2.line(
            image,
            (center_x, margin),
            (center_x, self.pitch_height - margin),
            white,
            thickness
        )

        # Center circle and spot
        cv2.circle(
            image,
            (center_x, center_y),
            65,
            white,
            thickness
        )

        cv2.circle(
            image,
            (center_x, center_y),
            3,
            white,
            -1
        )

        # Penalty boxes
        box_depth = 115
        box_height = 210

        cv2.rectangle(
            image,
            (margin, center_y - box_height // 2),
            (margin + box_depth, center_y + box_height // 2),
            white,
            thickness
        )

        cv2.rectangle(
            image,
            (self.pitch_width - margin - box_depth, center_y - box_height // 2),
            (self.pitch_width - margin, center_y + box_height // 2),
            white,
            thickness
        )

        # Goal areas
        goal_depth = 45
        goal_height = 100

        cv2.rectangle(
            image,
            (margin, center_y - goal_height // 2),
            (margin + goal_depth, center_y + goal_height // 2),
            white,
            thickness
        )

        cv2.rectangle(
            image,
            (self.pitch_width - margin - goal_depth, center_y - goal_height // 2),
            (self.pitch_width - margin, center_y + goal_height // 2),
            white,
            thickness
        )

        # Goals
        cv2.rectangle(
            image,
            (margin - 8, center_y - 35),
            (margin, center_y + 35),
            white,
            thickness
        )

        cv2.rectangle(
            image,
            (self.pitch_width - margin, center_y - 35),
            (self.pitch_width - margin + 8, center_y + 35),
            white,
            thickness
        )

        return image

    # --------------------------------------------------
    # STATIC HEATMAP SAVE
    # --------------------------------------------------

    def generate_heatmap(self, output_path, team=None):

        team_id = self.normalize_team(team)

        if team_id == 0:
            positions = self.team_a_positions

        elif team_id == 1:
            positions = self.team_b_positions

        else:
            positions = self.combined_positions

        image = self.render_map(positions)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        return cv2.imwrite(str(output_path), image)

    # --------------------------------------------------
    # VIDEO HEATMAP OVER ACTUAL FOOTBALL VIDEO
    # --------------------------------------------------

    def generate_overlay_video(
        self,
        input_video,
        output_path,
        team=None,
        trail_radius=35,
        opacity=0.55
    ):

        team_id = self.normalize_team(team)

        cap = cv2.VideoCapture(str(input_video))

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_video}")

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 25

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if width <= 0 or height <= 0:
            cap.release()
            raise RuntimeError("Invalid video dimensions")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Cannot create heatmap output video")

        # Cumulative heatmap buffer on original video resolution
        heat = np.zeros(
            (height, width),
            dtype=np.float32
        )

        # Group player positions by frame
        frame_positions = {}

        for frame_idx, x, y, player_team in self.position_history:

            if team_id is not None and player_team != team_id:
                continue

            frame_positions.setdefault(frame_idx, []).append((x, y))

        frame_idx = 0

        while True:

            success, frame = cap.read()

            if not success:
                break

            # Add player positions for this frame
            current_positions = frame_positions.get(frame_idx, [])

            for x, y in current_positions:

                x = max(0, min(width - 1, int(x)))
                y = max(0, min(height - 1, int(y)))

                cv2.circle(
                    heat,
                    (x, y),
                    trail_radius,
                    1.0,
                    -1
                )

            # Smooth cumulative heat
            blurred = cv2.GaussianBlur(
                heat,
                (0, 0),
                sigmaX=30,
                sigmaY=30
            )

            if blurred.max() > 0:

                normalized = np.clip(
                    blurred / max(float(blurred.max()), 1e-6),
                    0,
                    1
                )

                heat_uint8 = (
                    normalized * 255
                ).astype(np.uint8)

                heat_color = cv2.applyColorMap(
                    heat_uint8,
                    cv2.COLORMAP_JET
                )

                mask = heat_uint8 > 20

                # Overlay heatmap on original video
                frame[mask] = cv2.addWeighted(
                    frame[mask],
                    1.0 - opacity,
                    heat_color[mask],
                    opacity,
                    0
                )

            # Progress text
            if total_frames > 0:

                progress = min(
                    100,
                    int((frame_idx / total_frames) * 100)
                )

                cv2.putText(
                    frame,
                    f"HEATMAP {progress}%",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

            writer.write(frame)

            frame_idx += 1

        cap.release()
        writer.release()

        return str(output_path)

    # --------------------------------------------------
    # GET POSITION COUNT
    # --------------------------------------------------

    def get_position_count(self, team=None):

        team_id = self.normalize_team(team)

        if team_id == 0:
            return self.team_a_count

        elif team_id == 1:
            return self.team_b_count

        return self.combined_count

    # --------------------------------------------------
    # GET POSITION MAP
    # --------------------------------------------------

    def get_map(self, team=None):

        team_id = self.normalize_team(team)

        if team_id == 0:
            return self.team_a_positions

        elif team_id == 1:
            return self.team_b_positions

        return self.combined_positions

    # --------------------------------------------------
    # RENDER HEATMAP
    # --------------------------------------------------

    def render_heatmap(self, team=None):

        positions = self.get_map(team)

        return self.render_map(positions)

    # --------------------------------------------------
    # RESET
    # --------------------------------------------------

    def reset(self):

        self.position_history.clear()

        self.combined_positions.clear()
        self.team_a_positions.clear()
        self.team_b_positions.clear()

        self.combined_count = 0
        self.team_a_count = 0
        self.team_b_count = 0

        self.last_frame_index = 0
