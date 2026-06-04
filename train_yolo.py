from ultralytics import YOLO

def main():
    # Load a pretrained model (YOLOv8 Nano)
    model = YOLO('yolov8n.pt')

    # Train the model
    # Note: imgsz=320 is used because we downscale our camera frame for speed
    print("Starting YOLOv8 training on dataset.yaml...")
    results = model.train(
        data='dataset.yaml', 
        epochs=50, 
        imgsz=640, 
        device='cpu' # Change to '0' if you have an NVIDIA GPU installed
    )
    
    print("Training finished! Best model weights are usually saved in runs/detect/train/weights/best.pt")

if __name__ == '__main__':
    main()
