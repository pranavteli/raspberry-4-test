from ultralytics import YOLO

model = YOLO("Vending-YOLOv8n.onnx")

model.predict("test.jpg", save=True, conf=0.5)
