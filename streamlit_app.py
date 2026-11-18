
import csv
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import imageio_ffmpeg
import streamlit as st

from utils.player_tracker import FootballTracker
from utils.heatmap import FootballHeatmap


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_DIR / "input_videos"
OUTPUT_DIR = PROJECT_DIR / "output_videos"
DATA_DIR = PROJECT_DIR / "output_data"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# OUTPUT VIDEO PATHS
# ============================================================

ANNOTATED_RAW = OUTPUT_DIR / "football_analysis_raw.mp4"
ANNOTATED_VIDEO = OUTPUT_DIR / "football_analysis.mp4"

# Actual football video heatmap overlays
COMBINED_OVERLAY_RAW = OUTPUT_DIR / "combined_overlay_raw.mp4"
TEAM_A_OVERLAY_RAW = OUTPUT_DIR / "team_a_overlay_raw.mp4"
TEAM_B_OVERLAY_RAW = OUTPUT_DIR / "team_b_overlay_raw.mp4"

COMBINED_OVERLAY = OUTPUT_DIR / "combined_heatmap_overlay.mp4"
TEAM_A_OVERLAY = OUTPUT_DIR / "team_a_heatmap_overlay.mp4"
TEAM_B_OVERLAY = OUTPUT_DIR / "team_b_heatmap_overlay.mp4"

# Tactical pitch heatmap images
COMBINED_IMAGE = OUTPUT_DIR / "combined_heatmap.png"
TEAM_A_IMAGE = OUTPUT_DIR / "team_a_heatmap.png"
TEAM_B_IMAGE = OUTPUT_DIR / "team_b_heatmap.png"

# Tactical pitch heatmap animation raw videos
COMBINED_PITCH_RAW = OUTPUT_DIR / "combined_pitch_heatmap_raw.mp4"
TEAM_A_PITCH_RAW = OUTPUT_DIR / "team_a_pitch_heatmap_raw.mp4"
TEAM_B_PITCH_RAW = OUTPUT_DIR / "team_b_pitch_heatmap_raw.mp4"

# Tactical pitch heatmap animation final videos
COMBINED_PITCH_VIDEO = OUTPUT_DIR / "combined_pitch_heatmap_video.mp4"
TEAM_A_PITCH_VIDEO = OUTPUT_DIR / "team_a_pitch_heatmap_video.mp4"
TEAM_B_PITCH_VIDEO = OUTPUT_DIR / "team_b_pitch_heatmap_video.mp4"

CSV_PATH = DATA_DIR / "tracking_data.csv"


# ============================================================
# DETECTION HELPERS
# ============================================================

def get_value(detection, *keys, default=None):

    for key in keys:

        if isinstance(detection, dict) and key in detection:
            value = detection[key]

            if value is not None:
                return value

        elif hasattr(detection, key):
            value = getattr(detection, key)

            if value is not None:
                return value

    return default


def get_detection_type(detection):

    value = get_value(
        detection,
        "object_type",
        "class_name",
        "class",
        "label",
        "name",
        default="other"
    )

    return str(value).strip().lower()


def classify_detection(object_type):

    name = str(object_type).lower()

    if "goalkeeper" in name or "keeper" in name:
        return "goalkeeper"

    if "referee" in name or "official" in name:
        return "referee"

    if "ball" in name:
        return "ball"

    if "player" in name or "person" in name:
        return "player"

    return "other"


def get_bbox(detection):

    bbox = get_value(
        detection,
        "bbox",
        "xyxy",
        "box",
        default=None
    )

    if bbox is None:
        return None

    try:

        if hasattr(bbox, "tolist"):
            bbox = bbox.tolist()

        if len(bbox) == 1 and hasattr(bbox[0], "__len__"):
            bbox = bbox[0]

        if len(bbox) < 4:
            return None

        return tuple(
            int(float(v))
            for v in bbox[:4]
        )

    except (TypeError, ValueError, IndexError):
        return None


def get_team(detection):

    return get_value(
        detection,
        "team",
        "team_name",
        default=""
    )


def normalize_team(team):

    if team is None:
        return None

    value = str(team).strip().lower()

    if value in ("0", "a", "team a", "team_a"):
        return 0

    if value in ("1", "b", "team b", "team_b"):
        return 1

    return None


# ============================================================
# ACTUAL MATCH VIDEO HEATMAP OVERLAY FUNCTIONS
# ============================================================

def add_heat_point(heat, x, y, radius=24, strength=0.30):

    height, width = heat.shape

    x = int(x)
    y = int(y)

    if x < 0 or y < 0 or x >= width or y >= height:
        return

    cv2.circle(
        heat,
        (x, y),
        radius,
        strength,
        -1
    )


def create_heat_overlay(frame, heat, opacity=0.50):

    if heat.max() <= 0:
        return frame

    blurred = cv2.GaussianBlur(
        heat,
        (0, 0),
        sigmaX=28,
        sigmaY=28
    )

    normalized = blurred / max(float(blurred.max()), 1e-6)
    normalized = np.clip(normalized, 0, 1)

    heat_uint8 = (normalized * 255).astype(np.uint8)

    colored = cv2.applyColorMap(
        heat_uint8,
        cv2.COLORMAP_JET
    )

    mask = heat_uint8 > 18

    result = frame.copy()

    if np.any(mask):

        result[mask] = cv2.addWeighted(
            frame[mask],
            1.0 - opacity,
            colored[mask],
            opacity,
            0
        )

    return result


def write_overlay_frame(
    frame,
    combined_heat,
    team_a_heat,
    team_b_heat,
    combined_writer,
    team_a_writer,
    team_b_writer
):

    combined_frame = create_heat_overlay(
        frame,
        combined_heat
    )

    team_a_frame = create_heat_overlay(
        frame,
        team_a_heat
    )

    team_b_frame = create_heat_overlay(
        frame,
        team_b_heat
    )

    combined_writer.write(combined_frame)
    team_a_writer.write(team_a_frame)
    team_b_writer.write(team_b_frame)


# ============================================================
# VIDEO CONVERSION FOR STREAMLIT PLAYBACK
# ============================================================

def convert_to_h264(input_path, output_path):

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg,
        "-y",
        "-i", str(input_path),
        "-c:v", "libx264",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        str(output_path)
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2500:])

    return output_path


# ============================================================
# ANIMATED TACTICAL PITCH HEATMAP VIDEO
# ============================================================

def generate_pitch_heatmap_video(
    heatmap_object,
    output_raw,
    output_final,
    fps,
    total_frames,
    frame_width,
    frame_height,
    team=None
):

    pitch_width = heatmap_object.pitch_width
    pitch_height = heatmap_object.pitch_height

    output_raw = Path(output_raw)
    output_final = Path(output_final)

    output_raw.parent.mkdir(parents=True, exist_ok=True)
    output_final.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_raw),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (pitch_width, pitch_height)
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Cannot create tactical pitch video: {output_raw}"
        )

    team_id = heatmap_object.normalize_team(team)

    # Group player positions by video frame
    frame_positions = {}

    for frame_index, x, y, player_team in heatmap_object.position_history:

        if team_id is not None and player_team != team_id:
            continue

        pitch_x = int((x / frame_width) * pitch_width)
        pitch_y = int((y / frame_height) * pitch_height)

        pitch_x = max(0, min(pitch_width - 1, pitch_x))
        pitch_y = max(0, min(pitch_height - 1, pitch_y))

        frame_positions.setdefault(
            frame_index,
            []
        ).append((pitch_x, pitch_y))

    cumulative_positions = []

    try:

        for frame_index in range(1, total_frames + 1):

            # Positions recorded for the current frame
            current_positions = frame_positions.get(
                frame_index,
                []
            )

            cumulative_positions.extend(current_positions)

            # Render cumulative heatmap on football pitch
            pitch_frame = heatmap_object.render_map(
                cumulative_positions
            )

            # Add animation frame counter
            cv2.putText(
                pitch_frame,
                f"FRAME {frame_index}/{total_frames}",
                (35, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            writer.write(pitch_frame)

    finally:
        writer.release()

    # Convert animation for browser playback
    convert_to_h264(
        output_raw,
        output_final
    )

    return str(output_final)


# ============================================================
# ANALYZE FOOTBALL VIDEO
# ============================================================

def analyze_video(video_path, confidence):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("Cannot open input video.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Invalid video dimensions.")

    size = (width, height)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # --------------------------------------------------------
    # OUTPUT VIDEO WRITERS
    # --------------------------------------------------------

    annotated_writer = cv2.VideoWriter(
        str(ANNOTATED_RAW),
        fourcc,
        fps,
        size
    )

    combined_writer = cv2.VideoWriter(
        str(COMBINED_OVERLAY_RAW),
        fourcc,
        fps,
        size
    )

    team_a_writer = cv2.VideoWriter(
        str(TEAM_A_OVERLAY_RAW),
        fourcc,
        fps,
        size
    )

    team_b_writer = cv2.VideoWriter(
        str(TEAM_B_OVERLAY_RAW),
        fourcc,
        fps,
        size
    )

    writers = [
        annotated_writer,
        combined_writer,
        team_a_writer,
        team_b_writer
    ]

    if not all(writer.isOpened() for writer in writers):

        cap.release()

        for writer in writers:
            writer.release()

        raise RuntimeError(
            "Failed to initialize one or more video writers."
        )

    # --------------------------------------------------------
    # TRACKER + HEATMAP OBJECTS
    # --------------------------------------------------------

    tracker = FootballTracker()

    combined_heatmap = FootballHeatmap()
    team_a_heatmap = FootballHeatmap()
    team_b_heatmap = FootballHeatmap()

    # --------------------------------------------------------
    # VIDEO-RESOLUTION HEAT BUFFERS
    # --------------------------------------------------------

    combined_heat = np.zeros(
        (height, width),
        dtype=np.float32
    )

    team_a_heat = np.zeros(
        (height, width),
        dtype=np.float32
    )

    team_b_heat = np.zeros(
        (height, width),
        dtype=np.float32
    )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    csv_rows = []

    player_detections = 0
    referee_detections = 0
    goalkeeper_detections = 0
    ball_detections = 0

    team_a_track_ids = set()
    team_b_track_ids = set()

    frame_number = 0

    progress = st.progress(0.0)
    status = st.empty()

    # --------------------------------------------------------
    # FRAME PROCESSING
    # --------------------------------------------------------

    try:

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            detections = tracker.track(
                frame,
                confidence=confidence
            )

            if detections is None:
                detections = []

            annotated_frame = tracker.draw_detections(
                frame.copy(),
                detections
            )

            # ------------------------------------------------
            # PROCESS DETECTIONS
            # ------------------------------------------------

            for detection in detections:

                object_type = get_detection_type(detection)
                category = classify_detection(object_type)
                bbox = get_bbox(detection)
                team = get_team(detection)
                team_id = normalize_team(team)

                track_id = get_value(
                    detection,
                    "track_id",
                    "tracker_id",
                    "id",
                    default=None
                )

                confidence_value = get_value(
                    detection,
                    "confidence",
                    "conf",
                    default=""
                )

                # Detection statistics
                if category == "player":
                    player_detections += 1

                elif category == "referee":
                    referee_detections += 1

                elif category == "goalkeeper":
                    goalkeeper_detections += 1

                elif category == "ball":
                    ball_detections += 1

                if bbox is None:
                    continue

                x1, y1, x2, y2 = bbox

                # ------------------------------------------------
                # TRACKING IDS
                # ------------------------------------------------

                if track_id is not None:

                    if category == "player" and team_id == 0:
                        team_a_track_ids.add(track_id)

                    elif category == "player" and team_id == 1:
                        team_b_track_ids.add(track_id)

                # ------------------------------------------------
                # CSV RECORD
                # ------------------------------------------------

                csv_rows.append(
                    {
                        "frame": frame_number,
                        "track_id": track_id,
                        "object_type": object_type,
                        "team": team,
                        "confidence": confidence_value,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    }
                )

                # ------------------------------------------------
                # PLAYER POSITION (BOTTOM CENTER OF BBOX)
                # ------------------------------------------------

                if category != "player":
                    continue

                player_x = (x1 + x2) // 2
                player_y = y2

                player_x = max(0, min(width - 1, player_x))
                player_y = max(0, min(height - 1, player_y))

                # ------------------------------------------------
                # SCREEN-SPACE HEATMAPS ON ORIGINAL VIDEO
                # ------------------------------------------------

                add_heat_point(
                    combined_heat,
                    player_x,
                    player_y
                )

                if team_id == 0:

                    add_heat_point(
                        team_a_heat,
                        player_x,
                        player_y
                    )

                elif team_id == 1:

                    add_heat_point(
                        team_b_heat,
                        player_x,
                        player_y
                    )

                # ------------------------------------------------
                # TACTICAL PITCH POSITION COLLECTION
                # ------------------------------------------------

                combined_heatmap.add_position(
                    player_x,
                    player_y,
                    width,
                    height,
                    team=team_id,
                    frame_index=frame_number
                )

                if team_id == 0:

                    team_a_heatmap.add_position(
                        player_x,
                        player_y,
                        width,
                        height,
                        team=0,
                        frame_index=frame_number
                    )

                elif team_id == 1:

                    team_b_heatmap.add_position(
                        player_x,
                    player_y,
                    width,
                    height,
                    team=1,
                    frame_index=frame_number
                )

            # ----------------------------------------------------
            # WRITE VIDEO FRAMES
            # ----------------------------------------------------

            annotated_writer.write(annotated_frame)

            write_overlay_frame(
                frame,
                combined_heat,
                team_a_heat,
                team_b_heat,
                combined_writer,
                team_a_writer,
                team_b_writer
            )

            # ----------------------------------------------------
            # PROGRESS
            # ----------------------------------------------------

            if total_frames > 0:

                progress_value = min(
                    frame_number / total_frames,
                    1.0
                )

                progress.progress(progress_value)

            status.text(
                f"Processing frame {frame_number}"
                f" / {total_frames if total_frames > 0 else '?'}"
            )

    finally:

        cap.release()

        for writer in writers:
            writer.release()

    progress.progress(1.0)
    status.success("Football tracking completed.")

    # ============================================================
    # SAVE CSV
    # ============================================================

    fieldnames = [
        "frame",
        "track_id",
        "object_type",
        "team",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2"
    ]

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        csv_writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        csv_writer.writeheader()
        csv_writer.writerows(csv_rows)

    # ============================================================
    # GENERATE STATIC TACTICAL PITCH HEATMAP IMAGES
    # ============================================================

    status.info("Generating static tactical pitch heatmaps...")

    combined_heatmap.generate_heatmap(
        str(COMBINED_IMAGE)
    )

    team_a_heatmap.generate_heatmap(
        str(TEAM_A_IMAGE),
        team=0
    )

    team_b_heatmap.generate_heatmap(
        str(TEAM_B_IMAGE),
        team=1
    )

    # ============================================================
    # GENERATE ANIMATED TACTICAL PITCH HEATMAP VIDEOS
    # ============================================================

    status.info("Generating animated tactical pitch heatmaps...")

    generate_pitch_heatmap_video(
        combined_heatmap,
        COMBINED_PITCH_RAW,
        COMBINED_PITCH_VIDEO,
        fps,
        frame_number,
        width,
        height
    )

    generate_pitch_heatmap_video(
        team_a_heatmap,
        TEAM_A_PITCH_RAW,
        TEAM_A_PITCH_VIDEO,
        fps,
        frame_number,
        width,
        height,
        team=0
    )

    generate_pitch_heatmap_video(
        team_b_heatmap,
        TEAM_B_PITCH_RAW,
        TEAM_B_PITCH_VIDEO,
        fps,
        frame_number,
        width,
        height,
        team=1
    )

    status.success("Animated tactical pitch videos generated.")

    # ============================================================
    # CONVERT ACTUAL FOOTBALL VIDEOS TO H264
    # ============================================================

    status.info("Converting original football videos for browser playback...")

    convert_to_h264(
        ANNOTATED_RAW,
        ANNOTATED_VIDEO
    )

    convert_to_h264(
        COMBINED_OVERLAY_RAW,
        COMBINED_OVERLAY
    )

    convert_to_h264(
        TEAM_A_OVERLAY_RAW,
        TEAM_A_OVERLAY
    )

    convert_to_h264(
        TEAM_B_OVERLAY_RAW,
        TEAM_B_OVERLAY
    )

    status.success("Actual football videos converted successfully.")

    # ============================================================
    # RESULTS
    # ============================================================

    return {
        "annotated": str(ANNOTATED_VIDEO),

        "combined_overlay": str(COMBINED_OVERLAY),
        "team_a_overlay": str(TEAM_A_OVERLAY),
        "team_b_overlay": str(TEAM_B_OVERLAY),

        "combined_image": str(COMBINED_IMAGE),
        "team_a_image": str(TEAM_A_IMAGE),
        "team_b_image": str(TEAM_B_IMAGE),

        "combined_pitch_video": str(COMBINED_PITCH_VIDEO),
        "team_a_pitch_video": str(TEAM_A_PITCH_VIDEO),
        "team_b_pitch_video": str(TEAM_B_PITCH_VIDEO),

        "csv": str(CSV_PATH),

        "frames": frame_number,
        "players": player_detections,
        "referees": referee_detections,
        "goalkeepers": goalkeeper_detections,
        "balls": ball_detections,

        "team_a_count": len(team_a_track_ids),
        "team_b_count": len(team_b_track_ids)
    }


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="AI Football Video Analysis",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ AI Football Video Analysis System")

st.write(
    "Upload football footage to detect, track players, "
    "identify teams and generate video heatmap overlays."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Analysis Settings")

confidence = st.sidebar.slider(
    "Detection Confidence",
    min_value=0.10,
    max_value=0.90,
    value=0.30,
    step=0.05
)


# ============================================================
# VIDEO UPLOAD
# ============================================================

uploaded_video = st.file_uploader(
    "Upload Football Video",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded_video is not None:

    st.video(uploaded_video)

    if st.button(
        "Start Football Analysis",
        type="primary"
    ):

        temp_video_path = None

        try:

            suffix = Path(uploaded_video.name).suffix or ".mp4"

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_file:

                temp_file.write(
                    uploaded_video.getbuffer()
                )

                temp_video_path = temp_file.name

            with st.spinner(
                "Detecting players and generating heatmaps..."
            ):

                results = analyze_video(
                    temp_video_path,
                    confidence
                )

            st.session_state["football_results"] = results

            st.success("Football analysis finished!")

        except Exception as error:

            st.error("Analysis failed.")
            st.exception(error)

        finally:

            if temp_video_path and Path(temp_video_path).exists():

                try:
                    Path(temp_video_path).unlink()
                except OSError:
                    pass


# ============================================================
# DISPLAY RESULTS
# ============================================================

if "football_results" in st.session_state:

    results = st.session_state["football_results"]

    st.divider()
    st.header("📊 Analysis Results")

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    m1, m2, m3, m4, m5 = st.columns(5)

    m1.metric("Frames", results["frames"])
    m2.metric("Player Detections", results["players"])
    m3.metric("Referee Detections", results["referees"])
    m4.metric("Goalkeeper Detections", results["goalkeepers"])
    m5.metric("Ball Detections", results["balls"])

    # --------------------------------------------------------
    # ANNOTATED VIDEO
    # --------------------------------------------------------

    st.subheader("🎥 Annotated Football Video")

    annotated_path = Path(results["annotated"])

    if annotated_path.exists():

        annotated_bytes = annotated_path.read_bytes()

        st.video(
            annotated_bytes,
            format="video/mp4"
        )

        st.download_button(
            "Download Annotated Video",
            data=annotated_bytes,
            file_name="football_analysis.mp4",
            mime="video/mp4",
            key="download_annotated"
        )

    # --------------------------------------------------------
    # HEATMAP OVERLAY ON ACTUAL FOOTBALL VIDEO
    # --------------------------------------------------------

    st.divider()
    st.header("🔥 Heatmap Overlay on Actual Football Video")

    st.caption(
        "These videos show the original match footage with cumulative "
        "player-position heat added on top of the video."
    )

    combined_path = Path(results["combined_overlay"])

    if combined_path.exists():

        st.subheader("Combined Player Heatmap Overlay")

        combined_bytes = combined_path.read_bytes()

        st.video(
            combined_bytes,
            format="video/mp4"
        )

        st.download_button(
            "Download Combined Heatmap Overlay",
            data=combined_bytes,
            file_name="combined_heatmap_overlay.mp4",
            mime="video/mp4",
            key="download_combined_overlay"
        )

    col_a, col_b = st.columns(2)

    with col_a:

        st.subheader("Team A Heatmap Overlay")

        path_a = Path(results["team_a_overlay"])

        if path_a.exists():

            bytes_a = path_a.read_bytes()

            st.video(
                bytes_a,
                format="video/mp4"
            )

            st.download_button(
                "Download Team A Overlay",
                data=bytes_a,
                file_name="team_a_heatmap_overlay.mp4",
                mime="video/mp4",
                key="download_team_a_overlay"
            )

    with col_b:

        st.subheader("Team B Heatmap Overlay")

        path_b = Path(results["team_b_overlay"])

        if path_b.exists():

            bytes_b = path_b.read_bytes()

            st.video(
                bytes_b,
                format="video/mp4"
            )

            st.download_button(
                "Download Team B Overlay",
                data=bytes_b,
                file_name="team_b_heatmap_overlay.mp4",
                mime="video/mp4",
                key="download_team_b_overlay"
            )

    # --------------------------------------------------------
    # ANIMATED TACTICAL PITCH HEATMAP VIDEOS
    # --------------------------------------------------------

    st.divider()
    st.header("🎬 Animated Tactical Pitch Heatmaps")

    st.caption(
        "Watch player position heat build up progressively on the "
        "football pitch throughout the match."
    )

    anim_col1, anim_col2, anim_col3 = st.columns(3)

    animated_pitch_items = [
        (
            anim_col1,
            "Combined Pitch Heatmap Video",
            results["combined_pitch_video"],
            "download_combined_pitch_video"
        ),
        (
            anim_col2,
            "Team A Pitch Heatmap Video",
            results["team_a_pitch_video"],
            "download_team_a_pitch_video"
        ),
        (
            anim_col3,
            "Team B Pitch Heatmap Video",
            results["team_b_pitch_video"],
            "download_team_b_pitch_video"
        )
    ]

    for column, title, video_path, button_key in animated_pitch_items:

        with column:

            st.subheader(title)

            path = Path(video_path)

            if path.exists():

                video_bytes = path.read_bytes()

                st.video(
                    video_bytes,
                    format="video/mp4"
                )

                st.download_button(
                    f"Download {title}",
                    data=video_bytes,
                    file_name=path.name,
                    mime="video/mp4",
                    key=button_key
                )

            else:

                st.warning(
                    f"Video not found: {path.name}"
                )

    # --------------------------------------------------------
    # STATIC TACTICAL PITCH HEATMAP IMAGES
    # --------------------------------------------------------

    st.divider()
    st.header("⚽ Tactical Pitch Heatmaps")

    st.caption(
        "Static visualizations of player positions mapped onto a tactical pitch."
    )

    pitch_col1, pitch_col2, pitch_col3 = st.columns(3)

    image_items = [
        (
            pitch_col1,
            "Combined Player Heatmap",
            results["combined_image"]
        ),
        (
            pitch_col2,
            "Team A Heatmap",
            results["team_a_image"]
        ),
        (
            pitch_col3,
            "Team B Heatmap",
            results["team_b_image"]
        )
    ]

    for column, title, image_path in image_items:

        with column:

            st.subheader(title)

            path = Path(image_path)

            if path.exists():

                image_bytes = path.read_bytes()

                st.image(
                    image_bytes,
                    use_container_width=True
                )

                st.download_button(
                    f"Download {title}",
                    data=image_bytes,
                    file_name=path.name,
                    mime="image/png",
                    key=f"download_{path.name}"
                )

            else:

                st.warning(
                    f"Image not found: {path.name}"
                )

    # --------------------------------------------------------
    # TRACKING CSV
    # --------------------------------------------------------

    st.divider()
    st.header("📄 Tracking Data")

    csv_path = Path(results["csv"])

    if csv_path.exists():

        csv_bytes = csv_path.read_bytes()

        st.download_button(
            "Download Tracking CSV",
            data=csv_bytes,
            file_name="tracking_data.csv",
            mime="text/csv",
            key="download_tracking_csv"
        )

        with open(
            csv_path,
            "r",
            encoding="utf-8",
            newline=""
        ) as file:

            rows = list(
                csv.DictReader(file)
            )[:10]

        st.subheader("Tracking CSV Preview")

        for index, row in enumerate(rows, start=1):

            st.text(
                f"Record {index}: {row}"
            )
