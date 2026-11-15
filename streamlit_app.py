



import streamlit as st
import cv2
import csv
import subprocess
import imageio_ffmpeg
import numpy as np

from pathlib import Path
from collections import Counter

from utils.player_tracker import FootballTracker
from utils.heatmap import FootballHeatmap


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "input_videos"
OUTPUT_DIR = BASE_DIR / "output_videos"
DATA_DIR = BASE_DIR / "output_data"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

RAW_VIDEO_PATH = OUTPUT_DIR / "football_analysis_raw.mp4"
OUTPUT_VIDEO_PATH = OUTPUT_DIR / "football_analysis.mp4"

COMBINED_HEATMAP_VIDEO = OUTPUT_DIR / "combined_heatmap_video.mp4"
TEAM_A_HEATMAP_VIDEO = OUTPUT_DIR / "team_a_heatmap_video.mp4"
TEAM_B_HEATMAP_VIDEO = OUTPUT_DIR / "team_b_heatmap_video.mp4"

COMBINED_HEATMAP_IMAGE = OUTPUT_DIR / "combined_heatmap.png"
TEAM_A_HEATMAP_IMAGE = OUTPUT_DIR / "team_a_heatmap.png"
TEAM_B_HEATMAP_IMAGE = OUTPUT_DIR / "team_b_heatmap.png"

CSV_PATH = DATA_DIR / "tracking_data.csv"


# ============================================================
# VIDEO CONVERSION
# ============================================================

def convert_to_h264(input_path, output_path):
    """Convert OpenCV MP4 output to browser-compatible H264."""

    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg_path,
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
        raise RuntimeError(result.stderr[-2000:])

    return output_path


# ============================================================
# FOOTBALL VIDEO ANALYSIS
# ============================================================

def analyze_video(video_path, confidence):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("Could not open input football video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Invalid video dimensions.")

    video_fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(RAW_VIDEO_PATH),
        video_fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create annotated video writer.")

    # --------------------------------------------------------
    # HEATMAP INITIALIZATION
    # --------------------------------------------------------

    combined_heatmap = FootballHeatmap()
    team_a_heatmap = FootballHeatmap()
    team_b_heatmap = FootballHeatmap()

    heatmap_width = combined_heatmap.pitch_width
    heatmap_height = combined_heatmap.pitch_height

    heatmap_size = (
        heatmap_width,
        heatmap_height
    )

    heatmap_fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    combined_heatmap_writer = cv2.VideoWriter(
        str(COMBINED_HEATMAP_VIDEO),
        heatmap_fourcc,
        fps,
        heatmap_size
    )

    team_a_heatmap_writer = cv2.VideoWriter(
        str(TEAM_A_HEATMAP_VIDEO),
        heatmap_fourcc,
        fps,
        heatmap_size
    )

    team_b_heatmap_writer = cv2.VideoWriter(
        str(TEAM_B_HEATMAP_VIDEO),
        heatmap_fourcc,
        fps,
        heatmap_size
    )

    if not combined_heatmap_writer.isOpened():
        cap.release()
        writer.release()
        raise RuntimeError("Combined heatmap writer failed.")

    if not team_a_heatmap_writer.isOpened():
        cap.release()
        writer.release()
        combined_heatmap_writer.release()
        raise RuntimeError("Team A heatmap writer failed.")

    if not team_b_heatmap_writer.isOpened():
        cap.release()
        writer.release()
        combined_heatmap_writer.release()
        team_a_heatmap_writer.release()
        raise RuntimeError("Team B heatmap writer failed.")

    # --------------------------------------------------------
    # TRACKER
    # --------------------------------------------------------

    tracker = FootballTracker()

    tracking_rows = []

    team_a_ids = set()
    team_b_ids = set()
    referee_ids = set()
    goalkeeper_ids = set()

    object_counts = Counter()

    frame_number = 0

    progress = st.progress(0)
    status = st.empty()

    # --------------------------------------------------------
    # PROCESS VIDEO
    # --------------------------------------------------------

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        # YOLO + ByteTrack detection
        detections = tracker.track(
            frame,
            confidence
        )

        # Annotated match frame
        annotated_frame = tracker.draw_detections(
            frame.copy(),
            detections
        )

        # ----------------------------------------------------
        # DETECTIONS + PLAYER POSITIONS
        # ----------------------------------------------------

        for detection in detections:

            object_type = detection.get(
                "object_type",
                "other"
            )

            track_id = detection.get("track_id")

            team = detection.get("team")

            confidence_value = detection.get(
                "confidence",
                0
            )

            bbox = detection.get(
                "bbox",
                [0, 0, 0, 0]
            )

            class_name = detection.get(
                "class_name",
                object_type
            )

            object_counts[object_type] += 1

            if track_id is not None:

                if object_type == "player":

                    if team == 0:
                        team_a_ids.add(track_id)

                    elif team == 1:
                        team_b_ids.add(track_id)

                elif object_type == "referee":

                    referee_ids.add(track_id)

                elif object_type == "goalkeeper":

                    goalkeeper_ids.add(track_id)

            # -----------------------------------------------
            # PLAYER HEATMAP POSITIONS
            # -----------------------------------------------

            if object_type == "player":

                x1, y1, x2, y2 = map(
                    int,
                    bbox
                )

                # Player foot position
                player_x = (x1 + x2) // 2
                player_y = y2

                # Combined player activity
                combined_heatmap.add_position(
                    player_x,
                    player_y,
                    width,
                    height
                )

                # Team A activity
                if team == 0:

                    team_a_heatmap.add_position(
                        player_x,
                        player_y,
                        width,
                        height
                    )

                # Team B activity
                elif team == 1:

                    team_b_heatmap.add_position(
                        player_x,
                        player_y,
                        width,
                        height
                    )

            # -----------------------------------------------
            # TRACKING CSV
            # -----------------------------------------------

            tracking_rows.append({
                "frame": frame_number,
                "object_type": object_type,
                "track_id": track_id,
                "team": team,
                "confidence": confidence_value,
                "class_name": class_name,
                "x1": bbox[0],
                "y1": bbox[1],
                "x2": bbox[2],
                "y2": bbox[3]
            })

        # ----------------------------------------------------
        # WRITE ANNOTATED MATCH VIDEO
        # ----------------------------------------------------

        writer.write(annotated_frame)

        # ----------------------------------------------------
        # WRITE TACTICAL HEATMAP VIDEOS
        # ----------------------------------------------------

        combined_frame = combined_heatmap.render_frame()

        team_a_frame = team_a_heatmap.render_frame()

        team_b_frame = team_b_heatmap.render_frame()

        combined_heatmap_writer.write(
            combined_frame
        )

        team_a_heatmap_writer.write(
            team_a_frame
        )

        team_b_heatmap_writer.write(
            team_b_frame
        )

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if total_frames > 0:

            progress.progress(
                min(frame_number / total_frames, 1.0)
            )

        status.text(
            f"Processing frame {frame_number}/{total_frames}"
        )

    # --------------------------------------------------------
    # RELEASE VIDEO RESOURCES
    # --------------------------------------------------------

    cap.release()
    writer.release()

    combined_heatmap_writer.release()
    team_a_heatmap_writer.release()
    team_b_heatmap_writer.release()

    progress.progress(1.0)
    status.success("Video analysis completed.")

    # --------------------------------------------------------
    # SAVE FINAL STATIC HEATMAPS
    # --------------------------------------------------------

    combined_heatmap.generate_heatmap(
        str(COMBINED_HEATMAP_IMAGE)
    )

    team_a_heatmap.generate_heatmap(
        str(TEAM_A_HEATMAP_IMAGE)
    )

    team_b_heatmap.generate_heatmap(
        str(TEAM_B_HEATMAP_IMAGE)
    )

    # --------------------------------------------------------
    # SAVE TRACKING CSV
    # --------------------------------------------------------

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        fieldnames = [
            "frame",
            "object_type",
            "track_id",
            "team",
            "confidence",
            "class_name",
            "x1",
            "y1",
            "x2",
            "y2"
        ]

        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        csv_writer.writeheader()

        csv_writer.writerows(
            tracking_rows
        )

    # --------------------------------------------------------
    # CONVERT VIDEOS TO H264
    # --------------------------------------------------------

    status.text("Converting videos to H264...")

    convert_to_h264(
        RAW_VIDEO_PATH,
        OUTPUT_VIDEO_PATH
    )

    convert_to_h264(
        COMBINED_HEATMAP_VIDEO,
        OUTPUT_DIR / "combined_heatmap_h264.mp4"
    )

    convert_to_h264(
        TEAM_A_HEATMAP_VIDEO,
        OUTPUT_DIR / "team_a_heatmap_h264.mp4"
    )

    convert_to_h264(
        TEAM_B_HEATMAP_VIDEO,
        OUTPUT_DIR / "team_b_heatmap_h264.mp4"
    )

    status.success("All videos generated successfully!")

    return {
        "frames": frame_number,
        "fps": fps,
        "width": width,
        "height": height,
        "team_a": len(team_a_ids),
        "team_b": len(team_b_ids),
        "referees": len(referee_ids),
        "goalkeepers": len(goalkeeper_ids),
        "tracking_records": len(tracking_rows)
    }


# ============================================================
# STREAMLIT INTERFACE
# ============================================================

st.set_page_config(
    page_title="AI Football Video Analysis",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ AI Football Video Analysis System")

st.write(
    "Analyze football videos using AI player detection, "
    "tracking, team identification, and tactical heatmaps."
)

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

st.sidebar.header("Analysis Settings")

confidence = st.sidebar.slider(
    "Detection Confidence",
    min_value=0.10,
    max_value=0.90,
    value=0.30,
    step=0.05
)

# ------------------------------------------------------------
# VIDEO UPLOAD
# ------------------------------------------------------------

st.header("📹 Upload Football Video")

uploaded_video = st.file_uploader(
    "Select a football video",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded_video is not None:

    input_video_path = (
        INPUT_DIR / uploaded_video.name
    )

    with open(
        input_video_path,
        "wb"
    ) as video_file:

        video_file.write(
            uploaded_video.getbuffer()
        )

    st.success(
        f"Uploaded: {uploaded_video.name}"
    )

    st.video(
        uploaded_video.getvalue()
    )

    # --------------------------------------------------------
    # ANALYZE BUTTON
    # --------------------------------------------------------

    if st.button(
        "🚀 Start Football Analysis",
        type="primary"
    ):

        try:

            with st.spinner(
                "Analyzing football video..."
            ):

                results = analyze_video(
                    input_video_path,
                    confidence
                )

                st.session_state.analysis_results = results
                st.session_state.analysis_completed = True

        except Exception as error:

            st.error(
                f"Analysis failed: {error}"
            )

            st.exception(error)


# ============================================================
# ANALYSIS RESULTS
# ============================================================

if st.session_state.get(
    "analysis_completed",
    False
):

    results = st.session_state.analysis_results

    st.divider()

    st.header("📊 Analysis Statistics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Frames Processed",
        results["frames"]
    )

    col2.metric(
        "Team A Players",
        results["team_a"]
    )

    col3.metric(
        "Team B Players",
        results["team_b"]
    )

    col4.metric(
        "Tracking Records",
        results["tracking_records"]
    )

    col5, col6 = st.columns(2)

    col5.metric(
        "Referees",
        results["referees"]
    )

    col6.metric(
        "Goalkeepers",
        results["goalkeepers"]
    )

    # --------------------------------------------------------
    # ANNOTATED MATCH VIDEO
    # --------------------------------------------------------

    st.divider()

    st.header("🎥 Annotated Match Video")

    if OUTPUT_VIDEO_PATH.exists():

        st.video(
            OUTPUT_VIDEO_PATH.read_bytes(),
            format="video/mp4"
        )

        st.download_button(
            "Download Annotated Match Video",
            data=OUTPUT_VIDEO_PATH.read_bytes(),
            file_name="football_analysis.mp4",
            mime="video/mp4"
        )

    # --------------------------------------------------------
    # TACTICAL HEATMAP VIDEOS
    # --------------------------------------------------------

    st.divider()

    st.header("🔥 Tactical Heatmap Videos")

    st.caption(
        "Player positions accumulate over time. "
        "Red and yellow areas indicate higher activity."
    )

    combined_video = (
        OUTPUT_DIR / "combined_heatmap_h264.mp4"
    )

    team_a_video = (
        OUTPUT_DIR / "team_a_heatmap_h264.mp4"
    )

    team_b_video = (
        OUTPUT_DIR / "team_b_heatmap_h264.mp4"
    )

    # Combined heatmap
    st.subheader("Combined Player Activity")

    if combined_video.exists():

        st.video(
            combined_video.read_bytes(),
            format="video/mp4"
        )

        st.download_button(
            "Download Combined Tactical Heatmap",
            data=combined_video.read_bytes(),
            file_name="combined_heatmap_video.mp4",
            mime="video/mp4",
            key="download_combined_heatmap"
        )

    # Team A and Team B
    heatmap_col1, heatmap_col2 = st.columns(2)

    with heatmap_col1:

        st.subheader("Team A Activity")

        if team_a_video.exists():

            st.video(
                team_a_video.read_bytes(),
                format="video/mp4"
            )

            st.download_button(
                "Download Team A Heatmap",
                data=team_a_video.read_bytes(),
                file_name="team_a_heatmap_video.mp4",
                mime="video/mp4",
                key="download_team_a_heatmap"
            )

    with heatmap_col2:

        st.subheader("Team B Activity")

        if team_b_video.exists():

            st.video(
                team_b_video.read_bytes(),
                format="video/mp4"
            )

            st.download_button(
                "Download Team B Heatmap",
                data=team_b_video.read_bytes(),
                file_name="team_b_heatmap_video.mp4",
                mime="video/mp4",
                key="download_team_b_heatmap"
            )

    # --------------------------------------------------------
    # FINAL STATIC HEATMAPS
    # --------------------------------------------------------

    st.divider()

    st.header("🗺️ Final Tactical Heatmaps")

    image_col1, image_col2, image_col3 = st.columns(3)

    with image_col1:

        st.subheader("Combined")

        if COMBINED_HEATMAP_IMAGE.exists():

            st.image(
                str(COMBINED_HEATMAP_IMAGE),
                use_container_width=True
            )

    with image_col2:

        st.subheader("Team A")

        if TEAM_A_HEATMAP_IMAGE.exists():

            st.image(
                str(TEAM_A_HEATMAP_IMAGE),
                use_container_width=True
            )

    with image_col3:

        st.subheader("Team B")

        if TEAM_B_HEATMAP_IMAGE.exists():

            st.image(
                str(TEAM_B_HEATMAP_IMAGE),
                use_container_width=True
            )

    # --------------------------------------------------------
    # TRACKING DATA
    # --------------------------------------------------------

    st.divider()

    st.header("📄 Tracking Data")

    if CSV_PATH.exists():

        with open(
            CSV_PATH,
            "r",
            encoding="utf-8"
        ) as csv_file:

            csv_content = csv_file.read()

        st.download_button(
            "Download Tracking CSV",
            data=csv_content,
            file_name="tracking_data.csv",
            mime="text/csv"
        )

        st.text_area(
            "Tracking CSV Preview",
            value="\n".join(
                csv_content.splitlines()[:20]
            ),
            height=250
        )
