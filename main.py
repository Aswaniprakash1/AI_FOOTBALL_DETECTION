
from config import INPUT_VIDEO, OUTPUT_VIDEO, OUTPUT_CSV

from utils.video_utils import VideoHandler, VideoWriterHandler
from utils.player_tracker import FootballTracker
from utils.team_classifier import TeamClassifier
from utils.pitch_mapper import PitchMapper
from utils.heatmap import HeatmapGenerator
from utils.statistics import FootballStatistics
from utils.data_export import DataExporter


def main():

    print("Starting Football Analysis System")

    # 1. Open input video
    video = VideoHandler(INPUT_VIDEO)

    info = video.get_video_info()

    print("Video information:", info)

    # 2. Initialize YOLO tracker
    tracker = FootballTracker()

    # 3. Initialize other modules
    team_classifier = TeamClassifier()
    pitch_mapper = PitchMapper()

    heatmap = HeatmapGenerator()

    statistics = FootballStatistics(
        fps=info["fps"]
    )

    exporter = DataExporter(OUTPUT_CSV)

    # 4. Create output video writer
    writer = VideoWriterHandler(
        OUTPUT_VIDEO,
        info["fps"],
        info["width"],
        info["height"]
    )

    frame_number = 0

    try:

        while True:

            frame = video.read_frame()

            if frame is None:
                break

            frame_number += 1

            # 5. Detect and track objects
            detections = tracker.track(frame)

            # 6. Initialize team classifier
            if not team_classifier.initialized:
                team_classifier.fit(
                    frame,
                    detections
                )

            # 7. Assign team to players
            for detection in detections:

                if detection["object_type"] in [
                    "player",
                    "goalkeeper"
                ]:

                    team = team_classifier.predict_team(
                        frame,
                        detection["bbox"]
                    )

                    detection["team"] = team

                else:
                    detection["team"] = None

            # 8. Draw detections
            annotated_frame = tracker.draw_detections(
                frame.copy(),
                detections
            )

            # 9. Save tracking information
            for detection in detections:

                bbox = detection["bbox"]

                x1, y1, x2, y2 = bbox

                player_id = detection["track_id"]

                # Bottom-center of bounding box
                center_position = (
                    (x1 + x2) / 2,
                    y2
                )

                pitch_position = pitch_mapper.map_position(
                    center_position
                )

                if (
                    player_id is not None
                    and pitch_position is not None
                ):

                    heatmap.add_position(
                        player_id,
                        pitch_position
                    )

                    statistics.add_position(
                        player_id,
                        pitch_position,
                        frame_number
                    )

                exporter.add_detection(
                    frame_number=frame_number,
                    player_id=player_id,
                    object_type=detection["object_type"],
                    class_name=detection["class_name"],
                    bbox=bbox,
                    confidence=detection["confidence"],
                    team=detection.get("team"),
                    pitch_x=(
                        pitch_position[0]
                        if pitch_position else None
                    ),
                    pitch_y=(
                        pitch_position[1]
                        if pitch_position else None
                    )
                )

            # 10. Write annotated frame
            writer.write_frame(
                annotated_frame
            )

            if frame_number % 100 == 0:
                print(
                    f"Processed frames: {frame_number}"
                )

    finally:

        # 11. Release resources
        video.release()
        writer.release()

        exporter.save()

    print("Football analysis completed")
    print("Output video:", OUTPUT_VIDEO)
    print("Output CSV:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
