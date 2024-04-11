# import cv2
# from ultralytics import YOLO
# import numpy
# import sys
# # sys.path.append('..') 
# # from sort.sort import Sort
# coco_model = YOLO('yolov8n.pt')

# model = YOLO('../model/runs/detect/yolov8n_custom3/weights/best.pt')  # Adjust the model file as needed
# #/Users/tringuyen1803/Desktop/B351-Final-Project/model/runs/detect/yolov8n_custom3/weights/best.pt
# # Open the video
# cap = cv2.VideoCapture("helmet-video.mp4")
# results = {}
# frame_nmr = -1
# ret = True
# vehicles = [2, 3, 5, 7]
# # mot_tracker = Sort()
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
# size = (frame_width, frame_height)

# # Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# out = cv2.VideoWriter('./outputs/processed.avi', fourcc, 20.0, size)

# def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=6, line_length_x=200, line_length_y=200):
#     x1, y1 = top_left
#     x2, y2 = bottom_right

#     cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #-- top-left
#     cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

#     cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #-- bottom-left
#     cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

#     cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #-- top-right
#     cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

#     cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #-- bottom-right
#     cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

#     return img


# while ret:
#     frame_nmr += 1
#     ret, frame = cap.read()
#     if ret:
#         results[frame_nmr] = {}
#         # detect vehicles
#         detections = coco_model(frame)[0]
#         detections_ = []
#         for detection in detections.boxes.data.tolist():
#             x1, y1, x2, y2, score, class_id = detection
#             print(detection)
#             if int(class_id) in vehicles:
#                 detections_.append([x1, y1, x2, y2, score])
#         draw_border(
#                 frame, (int(x1), int(y1)),
#                 (int(x2), int(y2)), (0, 255, 0),
#                 12, line_length_x=200, line_length_y=200)
        
#         # track vehicles
#         #track_ids = mot_tracker.update(np.asarray(detections_))

#         # detect license plates
#         license_plates = model(frame)[0]
#         for license_plate in license_plates.boxes.data.tolist():
#             x1, y1, x2, y2, score, class_id = license_plate

#             print(license_plate)
            

            

#         out.write(frame)
#         frame = cv2.resize(frame, (1280, 720))

#         # Press 'q' to quit
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# # Release the video capture object and close the display window
# out.release()
# cap.release()
# cv2.destroyAllWindows()

import cv2
from ultralytics import YOLO
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import sys
sys.path.append('..') 
from predictor import HelmetPredictor
# Load the model
coco_model = YOLO('yolov8n.pt')
model = YOLO('../model/runs/detect/yolov8n_custom3/weights/best.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_rcnn = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=5)
model_rcnn.load_state_dict(torch.load('../tracker/model_weights.pth', map_location=device))
# Open the video
cap = cv2.VideoCapture("helmet-video.mp4")

# Check if video opened successfully
if not cap.isOpened():
    print("Error opening video file")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('./outputs/processed.avi', fourcc, 20.0, size)
class_names = {0: 'WithHelmet', 1: 'Without Helmet', 2: 'Unknown', 3: 'NumberPlate', 4: 'Rider'}
class_names_vehicles = {2: 'car',
 3: 'motorcycle',
 4: 'airplane',
 5: 'bus',
 6: 'train',
 7: 'truck'}
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Vehicle detection and drawing borders
        detections = coco_model(frame)[0]
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in [2, 3, 5, 7]:  # Assuming these are the vehicle class IDs
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = class_names_vehicles[int(class_id)]  # Get the label from class ID
        
                label_text = f"{label}: {score:.2f}"
                cv2.putText(frame, label_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # License plate detection
        license_plates = model(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            label = class_names[int(class_id)]

            # Prepare the text for the label and score
            text = f"{label}: {score:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.putText(frame, text, (int(x1), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        # predictor = HelmetPredictor('../tracker/model_weights.pth')
        # helmets = predictor.predict_and_draw_frame(frame)
        

        # Write the frame into the file 'output.avi'
        out.write(frame)

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
