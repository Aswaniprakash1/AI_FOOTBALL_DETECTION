
import cv2
from pathlib import Path


class VideoHandler:
    def __init__(self, video_path):
        self.video_path = str(video_path)

        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Unable to open video: {self.video_path}"
            )

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )
        self.height = int(
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )
        self.total_frames = int(
            self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

    def get_video_info(self):
        return {
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "total_frames": self.total_frames
        }

    def read_frame(self):
        success, frame = self.cap.read()

        if success:
            return frame

        return None

    def release(self):
        if self.cap is not None:
            self.cap.release()


class VideoWriterHandler:
    def __init__(self, output_path, fps, width, height):

        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        self.writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height)
        )

        if not self.writer.isOpened():
            raise RuntimeError(
                f"Unable to create output video: {output_path}"
            )

    def write_frame(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()
