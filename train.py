from ultralytics import YOLO
import argparse

def train_model(data_config, epochs):
    # Load a model
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(
        data=data_config, 
        epochs=epochs, 
        imgsz=640,
        plots=True,
        project="../models",
        name="traffic_custom"
    )

    print("Training complete.")
    print(f"Best model saved to: {results.save_dir}")

    # Export the model
    success = model.export(format="onnx")
    print(f"Model exported to ONNX: {success}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lab 02: Custom Model Training")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml file")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    args = parser.parse_args()

    # Note: You need a valid data.yaml and dataset structure to run this.
    # See README.md for instructions on how to prepare the dataset.
    # train_model(args.data, args.epochs)
    print("This is a boilerplate. To run training, ensure you have a dataset and uncomment the train_model call.")
