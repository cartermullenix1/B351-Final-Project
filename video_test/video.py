import cv2
from ultralytics import YOLO
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import sys
sys.path.append('..') 
from predictor import HelmetPredictor
import easyocr
import string


from core.config import settings
from db.session import engine
from db.base_class import Base
from db.session import SessionLocal
from sqlalchemy.orm import Session
from db.base_class import ImageData, LicensePlateData
import cv2
import logging
import numpy as np

# Load the model
coco_model = YOLO('yolov8n.pt')
model = YOLO('../model/runs/detect/yolov8n_custom4/weights/best.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model_rcnn = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=5)
#model_rcnn.load_state_dict(torch.load('../tracker/model_weights.pth', map_location=device))
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
class_names = {0: 'WithHelmet', 1: 'Without Helmet', 2: 'Rider', 3: 'NumberPlate', 4: 'Unknown'}
class_names_vehicles = {2: 'car',
 3: 'motorcycle',
 4: 'airplane',
 5: 'bus',
 6: 'train',
 7: 'truck'}

dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}
def license_complies_format(text):
    # True if the license plate complies with the format, False otherwise.
    if len(text) != 7:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
        return True
    else:
        return False

def format_license(text):
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_

reader = easyocr.Reader(['en'], gpu=False)

def read_license_plate(license_plate_crop):
    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')

        # verify that text is conform to a standard license plate
        if license_complies_format(text):
            # bring text into the default license plate format
            return format_license(text), score

    return None, None

# def create_tables():         
# 	Base.metadata.create_all(bind=engine)

# def create_license_plate_data(db: Session, image_name: str, plate_text: str) -> LicensePlateData:
#     license_plate_data = LicensePlateData(image_name=image_name, plate_text=plate_text)
#     db.add(license_plate_data)
#     db.commit()
#     db.refresh(license_plate_data)
#     return license_plate_data

# create_tables()

# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

while cap.isOpened():
    ret, frame = cap.read()
    if ret:       
        detections = coco_model.track(frame, persist=True)[0]
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, track_id, score, class_id = detection
            if int(class_id) in class_names_vehicles and score > 0.5:
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
            
            if class_id == 3:
                plate = frame[int(y1):int(y2), int(x1):int(x2)]

                plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
                # posterize
                _, plate_treshold = cv2.threshold(plate_gray, 64, 255, cv2.THRESH_BINARY_INV)

                detections = reader.readtext(plate_gray)
                print("Detection", detections)
                
                License_text = ""
                for detection in detections:
                    bbox, text, confidence = detection
                    text_x1, text_y1, text_x2, text_y2 = bbox
                    
                
                    License_text += text
                #db = get_db()
                print(License_text)
                #create_license_plate_data(db, "Image name", License_text)

                
                cv2.putText(frame, License_text, (int(x1), int(y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                # display the license plate and the output image
            else:

                cv2.putText(frame, label, (int(x1), int(y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
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
