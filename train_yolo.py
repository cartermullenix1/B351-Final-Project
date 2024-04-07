from ultralytics import YOLO

# model = YOLO("model/runs/detect/yolov8n_custom3/weights/best.pt")
# print(model)
# results = model["archive/val/images/new4.jpg"]

# results.show()

from ultralytics import YOLO

# Load the model
model = YOLO("model/runs/detect/yolov8n_custom3/weights/best.pt")
print(model)

# Assuming 'archive/val/images/new4.jpg' is the path to your image
image_path = "archive/val/images/new86.jpg"

# Perform inference
results = model.predict(image_path)

for result in results:
    print(result.boxes)
    print(result.probs)
    result.save("processed_yolo.jpg")
    result.show()

