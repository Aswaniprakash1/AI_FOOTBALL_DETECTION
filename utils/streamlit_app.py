

import streamlit as st
import cv2
import csv
import tempfile
import subprocess
import imageio_ffmpeg

from pathlib import Path
from utils.player_tracker import FootballTracker
from utils.heatmap import FootballHeatmap


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_VIDEO_DIR = BASE_DIR / "output_videos"
OUTPUT_DATA_DIR = BASE_DIR / "output_data"

OUTPUT_VIDEO_DIR.mkdir(exist_ok=True)
OUTPUT_DATA_DIR.mkdir(exist_ok=True)

OUTPUT_VIDEO = OUTPUT_VIDEO_DIR / "football_analysis.mp4"
OUTPUT_CSV = OUTPUT_DATA_DIR / "tracking_data.csv"

# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Football Video Analysis",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ AI Football Video Analysis System")

st.markdown(
    "Upload a football video to detect players, "
    "track movement, identify teams and generate heatmaps."
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

    if st.button("Start Football Analysis", type="primary"):

        # Save uploaded video temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as temp_file:

            temp_file.write(uploaded_video.getbuffer())
            input_video_path = temp_file.name

        cap = cv2.VideoCapture(input_video_path)

        if not cap.isOpened():
            st.error("Unable to open uploaded video.")
            st.stop()

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 30

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        st.info(
            f"Video: {width} × {height} | "
            f"FPS: {fps:.2f} | Frames: {total_frames}"
        )

        # ====================================================
        # INITIALIZE TRACKER
        # ====================================================

        tracker = FootballTracker()

        # ====================================================
        # INITIALIZE HEATMAPS
        # ====================================================

        combined_heatmap = FootballHeatmap()
        team_a_heatmap = FootballHeatmap()
        team_b_heatmap = FootballHeatmap()

        # ====================================================
        # VIDEO WRITER
        # ====================================================

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            str(OUTPUT_VIDEO),
            fourcc,
            fps,
            (width, height)
        )

        # ====================================================
        # CSV
        # ====================================================

        csv_rows = []

        # ====================================================
        # ANALYSIS COUNTERS
        # ====================================================

        frame_count = 0
        player_count = 0
        referee_count = 0
        goalkeeper_count = 0
        ball_count = 0

        progress_bar = st.progress(0)
        status_text = st.empty()

        # ====================================================
        # PROCESS VIDEO
        # ====================================================

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            detections = tracker.track(
                frame,
                confidence=confidence
            )

            annotated_frame = tracker.draw_detections(
                frame.copy(),
                detections
            )

            # -----------------------------------------------
            # PROCESS DETECTIONS
            # -----------------------------------------------

            # ================= PLAYER HEATMAP TRACKING =================

for detection in detections:

    bbox = detection.get("bbox")
    object_type = detection.get("object_type", "")
    team = detection.get("team", "")

    if bbox is None:
        continue

    # Player positions only
    if str(object_type).lower() != "player":
        continue

    x1, y1, x2, y2 = map(int, bbox)

    # Player foot position
    player_x = (x1 + x2) // 2
    player_y = y2

    # Save player position for heatmap
    combined_heatmap.add_position(
        player_x,
        player_y,
        frame.shape[1],
        frame.shape[0]
    )

    # Team A
    if str(team).lower() in ["team a", "team_a", "a", "0"]:
        team_a_heatmap.add_position(
            player_x,
            player_y,
            frame.shape[1],
            frame.shape[0]
        )

    # Team B
    elif str(team).lower() in ["team b", "team_b", "b", "1"]:
        team_b_heatmap.add_position(
            player_x,
            player_y,
            frame.shape[1],
            frame.shape[0]
        )
                # -------------------------------------------
                # CSV ROW
                # -------------------------------------------

                csv_rows.append({
                    "frame": frame_count,
                    "track_id": track_id,
                    "object_type": object_type,
                    "team": team,
                    "confidence": detection.get("confidence"),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                })

            # -----------------------------------------------
            # WRITE COMPLETE ANNOTATED FRAME
            # -----------------------------------------------

            writer.write(annotated_frame)

            # -----------------------------------------------
            # PROGRESS
            # -----------------------------------------------

            if total_frames > 0:

                progress = min(
                    frame_count / total_frames,
                    1.0
                )

                progress_bar.progress(progress)

                status_text.text(
                    f"Processing frame "
                    f"{frame_count}/{total_frames}"
                )

        # ====================================================
        # RELEASE VIDEO
        # ====================================================

        cap.release()
        writer.release()

        progress_bar.progress(1.0)
        status_text.success("Football video analysis completed!")

        # ====================================================
        # SAVE TRACKING CSV
        # ====================================================

        with open(
            OUTPUT_CSV,
            "w",
            newline="",
            encoding="utf-8"
        ) as csv_file:

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

            csv_writer = csv.DictWriter(
                csv_file,
                fieldnames=fieldnames
            )

            csv_writer.writeheader()
            csv_writer.writerows(csv_rows)

        # ====================================================
        # GENERATE HEATMAPS
        # ====================================================

        combined_path = combined_heatmap.generate_heatmap(
            str(OUTPUT_VIDEO_DIR / "combined_heatmap.png")
        )

        team_a_path = team_a_heatmap.generate_heatmap(
            str(OUTPUT_VIDEO_DIR / "team_a_heatmap.png")
        )

        team_b_path = team_b_heatmap.generate_heatmap(
            str(OUTPUT_VIDEO_DIR / "team_b_heatmap.png")
        )

        # ====================================================
        # SAVE SESSION RESULTS
        # ====================================================

        st.session_state["football_results"] = {
            "video": str(OUTPUT_VIDEO),
            "csv": str(OUTPUT_CSV),
            "combined": combined_path,
            "team_a": team_a_path,
            "team_b": team_b_path,
            "frames": frame_count,
            "players": player_count,
            "referees": referee_count,
            "goalkeepers": goalkeeper_count,
            "balls": ball_count
        }

# ============================================================
# DISPLAY RESULTS
# ============================================================

if "football_results" in st.session_state:

    results = st.session_state["football_results"]

    st.divider()
    st.header("📊 Football Analysis Results")

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Frames", results["frames"])
    col2.metric("Player Detections", results["players"])
    col3.metric("Referees", results["referees"])
    col4.metric("Goalkeepers", results["goalkeepers"])
    col5.metric("Ball Detections", results["balls"])

    # --------------------------------------------------------
    # ANNOTATED VIDEO
    # --------------------------------------------------------

    st.subheader("🎥 Complete Annotated Football Video")

    video_path = Path(results["video"])

    if video_path.exists():

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        converted_video = OUTPUT_VIDEO_DIR / "football_analysis_h264.mp4"

        command = [
            ffmpeg_exe,
            "-y",
            "-i", str(video_path),
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            str(converted_video)
        ]

        with st.spinner("Preparing video playback..."):

            subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )

        if converted_video.exists():

            st.video(
                converted_video.read_bytes(),
                format="video/mp4"
            )

    # --------------------------------------------------------
    # CSV DOWNLOAD
    # --------------------------------------------------------

    st.subheader("📁 Tracking Data")

    csv_path = Path(results["csv"])

    if csv_path.exists():

        csv_bytes = csv_path.read_bytes()

        st.download_button(
            "Download Tracking CSV",
            data=csv_bytes,
            file_name="football_tracking_data.csv",
            mime="text/csv"
        )

        with open(
            csv_path,
            "r",
            encoding="utf-8"
        ) as csv_file:

            preview_rows = list(
                csv.DictReader(csv_file)
            )[:10]

        st.write("Tracking Data Preview")

        for row in preview_rows:
            st.text(str(row))

    # --------------------------------------------------------
    # FOOTBALL HEATMAPS
    # --------------------------------------------------------

    st.divider()
    st.header("🔥 Football Pitch Heatmaps")

    st.info(
        "Heatmaps show approximate player movement density. "
        "Accurate pitch coordinates require camera calibration."
    )

    heatmap_col1, heatmap_col2, heatmap_col3 = st.columns(3)

    heatmaps = [
        (
            "Combined Player Heatmap",
            results["combined"],
            "combined_heatmap.png"
        ),
        (
            "Team A Heatmap",
            results["team_a"],
            "team_a_heatmap.png"
        ),
        (
            "Team B Heatmap",
            results["team_b"],
            "team_b_heatmap.png"
        )
    ]

    for column, heatmap in zip(
        [heatmap_col1, heatmap_col2, heatmap_col3],
        heatmaps
    ):

        title, image_path, download_name = heatmap

        with column:

            st.subheader(title)

            image_file = Path(image_path)

            if image_file.exists():

                image_bytes = image_file.read_bytes()

                st.image(
                    image_bytes,
                    use_container_width=True
                )

                st.download_button(
                    f"Download {title}",
                    data=image_bytes,
                    file_name=download_name,
                    mime="image/png",
                    key=download_name
                )
