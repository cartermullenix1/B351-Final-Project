from ultralytics import YOLO

model = YOLO('yolov8n.pt')
 
# Training.
results = model.train(
   data='../archive/helmet_plate.yaml',
   imgsz=640,
   epochs=100,
   batch=8,
   name='yolov8n_custom'
)