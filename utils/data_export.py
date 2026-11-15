
import csv
from pathlib import Path


class DataExporter:
    def __init__(self, output_path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.data = []

    def add_detection(
        self,
        frame_number,
        player_id,
        object_type,
        class_name,
        bbox,
        confidence,
        team=None,
        pitch_x=None,
        pitch_y=None
    ):
        x1, y1, x2, y2 = bbox

        self.data.append({
            "frame": frame_number,
            "player_id": player_id,
            "object_type": object_type,
            "class_name": class_name,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "confidence": confidence,
            "team": team,
            "pitch_x": pitch_x,
            "pitch_y": pitch_y
        })

    def save(self):
        if not self.data:
            print("No tracking data available.")
            return

        with open(
            self.output_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=self.data[0].keys()
            )
            writer.writeheader()
            writer.writerows(self.data)

        print(f"Tracking data saved: {self.output_path}")

    def clear(self):
        self.data.clear()
