from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model(source="test.mp4",conf=0.4,save=True)
print(f"\n\n\n\n\{results}\n\n\n")