
from pathlib import Path

# ================= PROJECT PATHS =================

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_DIR / "dataset"
INPUT_VIDEO_DIR = PROJECT_DIR / "input_videos"
OUTPUT_VIDEO_DIR = PROJECT_DIR / "output_videos"
OUTPUT_DATA_DIR = PROJECT_DIR / "output_data"
MODEL_DIR = PROJECT_DIR / "models"

# ================= VIDEO FILES =================

INPUT_VIDEO = INPUT_VIDEO_DIR / "match.mp4"
OUTPUT_VIDEO = OUTPUT_VIDEO_DIR / "football_analysis.mp4"
OUTPUT_CSV = OUTPUT_DATA_DIR / "tracking_data.csv"

# ================= YOLO MODEL =================

MODEL_PATH = MODEL_DIR / "football_model.pt"

CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50

TRACKER_CONFIG = "bytetrack.yaml"

# ================= TEAM COLORS (BGR) =================

TEAM_A_COLOR = (255, 0, 0)       # Blue
TEAM_B_COLOR = (255, 255, 255)   # White
REFEREE_COLOR = (0, 255, 255)   # Yellow
BALL_COLOR = (0, 165, 255)       # Orange

# ================= PROCESSING =================

FRAME_SKIP = 0

# ================= FOOTBALL PITCH =================

PITCH_WIDTH = 105
PITCH_HEIGHT = 68
