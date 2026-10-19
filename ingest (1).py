import cv2
import argparse

def main(source):
    # Initialize video capture
    # source can be an integer (webcam index) or a string (video file path / RTSP url)
    # Try converting to int if it looks like a number (webcam)
    try:
        source = int(source)
    except ValueError:
        pass

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return

    print(f"Processing video from: {source}")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream or error reading frame.")
            break

        # Display the frame
        cv2.imshow("Lab 01 - Ingestion", frame)

        # Handle key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lab 01: Video Ingestion")
    parser.add_argument("--source", type=str, default="../data/sample.mp4", help="Path to video file or webcam index")
    args = parser.parse_args()

    main(args.source)
