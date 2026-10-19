import cv2
from ultralytics import YOLO
import argparse
import supervision as sv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def main(source, model_path):
    # Load the model (use your custom trained model from Lab 02 or a standard one)
    model = YOLO(model_path)

    # Initialize video capture
    cap = cv2.VideoCapture(source) if not source.isdigit() else cv2.VideoCapture(int(source))
    if not cap.isOpened():
        print("Error opening video source")
        return

    # Initialize annotators
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Inference
        results = model(frame)[0]

        # 2. Convert to Supervision Detections
        detections = sv.Detections.from_ultralytics(results)

        # 3. Filter (optional) - e.g. only cars
        # detections = detections[detections.class_id == 2] # 2 is car in COCO

        # 4. Annotate
        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, 
            detections=detections,
            labels=[f"{model.model.names[class_id]} {confidence:.2f}" for class_id, confidence in zip(detections.class_id, detections.confidence)]
        )

        cv2.imshow("Lab 03 - Object Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lab 03: Object Detection")
    parser.add_argument("--source", type=str, default=str(BASE_DIR / "data" / "traffic.mp4"), help="Path to video file")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to model weights")
    args = parser.parse_args()

    main(args.source, args.model)
