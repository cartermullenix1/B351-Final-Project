# from ultralytics import YOLO

# # model = YOLO("model/runs/detect/yolov8n_custom3/weights/best.pt")
# # print(model)
# # results = model["archive/val/images/new4.jpg"]

# # results.show()

from ultralytics import YOLO
from easyocr import Reader
import cv2
import string

# # Load the model
model = YOLO("model/runs/detect/yolov8n_custom4/weights/best.pt")
# print(model)

# # Assuming 'archive/val/images/new4.jpg' is the path to your image
# image_path = "archive/train/images/new40.jpg"
image_path = "archive/train/images/new39.jpg"
# # Perform inference
results = model.predict(image_path)

# for result in results:
#     print("Boxess", result.names)

#     result.save("processed_yolo.jpg")
#     result.show()
reader = Reader(['en'], gpu=False)

# license_plates = model.predict(image_path)[0]
# for license_plate in license_plates:
#     x1, y1, x2, y2, score, class_id = license_plate.data
#     cv2.rectangle(license_plates, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    
#     # label = class_names[int(class_id)]

#     if class_id == 3:
#         plate = license_plates[int(y1):int(y2), int(x1):int(x2)]

#                 # de-colorize
#         plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
#         # posterize
#         _, plate_treshold = cv2.threshold(plate_gray, 64, 255, cv2.THRESH_BINARY_INV)

#         cv2.imshow("threshold", plate_treshold)
#         cv2.imshow("threshold", plate)
#         cv2.waitKey(0)
#     # # OCR
#     # np_text, np_score = read_license_plate(plate_treshold)
#     # license_text = f"{np_text}"
#     # cv2.putText(frame, license_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)



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

reader = Reader(['en'], gpu=False)

def read_license_plate(license_plate_crop):
    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')

        # verify that text is conform to a standard license plate
        if license_complies_format(text):
            # bring text into the default license plate format
            return bbox,format_license(text), score

    return None, None

img = cv2.imread(image_path)

license_plates = model.predict(image_path)[0]
for license_plate in license_plates.boxes.data.tolist():
    print("License plate", license_plate)
    x1, y1, x2, y2, score, class_id = license_plate
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    print("Class id" ,class_id)


    #label = class_names[int(class_id)]
    
    if class_id == 3:
        plate = img[int(y1):int(y2), int(x1):int(x2)]

                # de-colorize
        plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        # posterize
        _, plate_treshold = cv2.threshold(plate_gray, 64, 255, cv2.THRESH_BINARY_INV)

        cv2.imshow("threshold", plate_treshold)
        cv2.imshow("threshold", plate)
        cv2.waitKey(0)

        detections = reader.readtext(plate_gray)
        print("Detection", detections)

        License_text = ""
        for detection in detections:
            bbox, text, confidence = detection
            text_x1, text_y1, text_x2, text_y2 = bbox
            print("Text x1", text_x1)
            
    
            License_text += text
        print(License_text)
        img = img.copy()
        print("X1", x1)
        print("Y1", y1)

        cv2.putText(img, License_text, (int(x1), int(y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        # display the license plate and the output image
        
        cv2.imshow('Image', img)
        #img.save("License detection")
        cv2.waitKey(0)

