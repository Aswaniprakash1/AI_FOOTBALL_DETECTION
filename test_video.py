

from ultralytics import YOLO
from config import MODEL_PATH


def main():

    print("Testing Football YOLO Model")

    if not MODEL_PATH.exists():
        print("ERROR: football_model.pt not found")
        return

    model = YOLO(str(MODEL_PATH))

    print("\nModel loaded successfully!")
    print("\nFootball Detection Classes:")

    for class_id, class_name in model.names.items():
        print(class_id, ":", class_name)

    print("\nFootball YOLO model test completed")


if __name__ == "__main__":
    main()
