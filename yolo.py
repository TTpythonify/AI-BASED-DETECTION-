from ultralytics import YOLO

# Load pretrained YOLOv8 model
model = YOLO("yolov8n.pt")   # Small & fast version

# Run detection on video
results = model(f"C:/Users/chidu/Downloads/IMG_3316.MOV", save=True, conf=0.4)

print("Processing complete! Check the 'runs/detect' folder.")
