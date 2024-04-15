import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import cv2
import easyocr
import string
import numpy as np

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

class HelmetPredictor:
    def __init__(self, model_path, num_classes=5, model_type='fasterrcnn'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_type == 'fasterrcnn':
            self.model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        elif model_type == 'yolov8':
            # Load YOLOv8 model
            #self.model = torch.hub.load('ultralytics/yolov8', 'custom', path=model_path, source='local')  # Ensure path and source are correct
            self.model = YOLO("model/runs/detect/yolov8n_custom3/weights/best.pt")
        else:
            raise ValueError("Unsupported model type: " + model_type)

        self.model.to(self.device).eval()
        self.class_names = {0: 'WithHelmet', 1: 'Without Helmet', 2: 'Rider', 3: 'NumberPlate', 4: 'Unknown'}

    def transform_image(self, image):
        if hasattr(self, 'model') and isinstance(self.model, torch.nn.Module):
            # Transform for Fast R-CNN
            image = F.to_tensor(image)
            return image.unsqueeze(0)  # Add a batch dimension
        else:
            # No transformation needed for YOLOv8 as it handles raw PIL images
            return image

    def predict(self, image):
        image_tensor = self.transform_image(image)
        with torch.no_grad():
            output = self.model(image_tensor.to(self.device))
        return output

    # def draw_boxes(self, image, predictions, threshold=0.5):
    #     draw = ImageDraw.Draw(image)
    #     font = ImageFont.load_default()
        
    #     for element in range(len(predictions[0]['boxes'])):
    #         score = predictions[0]['scores'][element] if 'scores' in predictions[0] else predictions[0]['confidences'][element]
    #         if score > threshold:
    #             box = predictions[0]['boxes'][element].tolist() if 'boxes' in predictions[0] else predictions[0]['xyxy'][element].tolist()
    #             label_idx = predictions[0]['labels'][element].item() if 'labels' in predictions[0] else predictions[0]['classes'][element].item()
    #             label = self.class_names.get(label_idx, 'Unknown')
    #             print("box", box)
    #             x1, y1, x2, y2 = box
    #             if label_idx == 3:
                    
    #                 plate = frame[int(y1):int(y2), int(x1):int(x2)]

    #                     # de-colorize
    #                 plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    #                 # posterize
    #                 _, plate_treshold = cv2.threshold(plate_gray, 64, 255, cv2.THRESH_BINARY_INV)

    #                 cv2.imshow("threshold", plate_treshold)
    #                 cv2.imshow("threshold", plate)
    #                 cv2.waitKey(0)
    #                 # OCR
    #                 np_text, np_score = read_license_plate(plate_treshold)
    #                 license_text = f"{np_text}"
    #                 cv2.putText(frame, license_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    #             draw.rectangle(box, outline='red', width=3)
    #             text = f"{label}: {score:.2f}"
    #             text_size = ImageFont.truetype('arial.ttf', 16) #draw.textsize(text, font=font)
    #             text_background = [box[0], box[1], box[0] + text_size[0], box[1] + text_size[1]]
    #             draw.rectangle(text_background, fill='red')
    #             draw.text((box[0], box[1]), text, fill='white', font=font)

    #     return image

    def textsize(text, font):
        im = Image.new(mode="P", size=(0, 0))
        draw = ImageDraw.Draw(im)
        _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
        return width, height
    
    def draw_boxes(self, image, predictions, threshold=0.5):
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        for element in range(len(predictions[0]['boxes'])):
            score = predictions[0]['scores'][element] if 'scores' in predictions[0] else predictions[0]['confidences'][element]
            if score > threshold:
                box = predictions[0]['boxes'][element].tolist()
                label_idx = predictions[0]['labels'][element].item()
                label = self.class_names.get(label_idx, 'Unknown')
                if label_idx == 3:  # Assuming '3' is the label for license plates
                    # Extracting the coordinates for the bounding box
                    x1, y1, x2, y2 = box
                    # Convert PIL Image to OpenCV format
                    frame = np.array(image)
                    # Extract the license plate region from the frame
                    plate = frame[int(y1):int(y2), int(x1):int(x2)]

                    # Convert to grayscale
                    plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
                    # Apply threshold to get binary image
                    _, plate_threshold = cv2.threshold(plate_gray, 64, 255, cv2.THRESH_BINARY_INV)

                    np_text, np_score = read_license_plate(plate_threshold)
                    license_text = f"{np_text}"
                    print("license", license_text)
                

                    # Convert OpenCV format back to PIL for consistency in drawing
                    image = Image.fromarray(frame.astype('uint8'), 'RGB')
                    draw = ImageDraw.Draw(image)

                # Draw rectangle around the detected license plate
                # draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
                # label = self.class_names.get(label_idx, 'Unknown')
                # text = f"{label}: {score:.2f}"
                # text_size = draw.textsize(text, font=font)
                # text_background = [x1, y1 - text_size[1], x1 + text_size[0], y1]
                # draw.rectangle(text_background, fill='red')
                # draw.text((x1, y1 - text_size[1]), text, fill='white', font=font)
                draw.rectangle(box, outline='red', width=3)
                text = f"{label}: {score:.2f}"
                
                text_size = draw.textsize(text, font=font)
                text_background = [box[0], box[1], box[0] + text_size[0], box[1] + text_size[1]]
                draw.rectangle(text_background, fill='red')
                draw.text((box[0], box[1]), text, fill='white', font=font)

        return image

    def predict_and_draw(self, image_path, threshold=0.5):
        image = Image.open(image_path).convert('RGB')
        predictions = self.predict(image)
        return self.draw_boxes(image, predictions, threshold)
    
    # def predict_and_draw_frame(self, frame, threshold=0.2):
    #     # Convert frame to PIL image
    #     pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    #     font = ImageFont.load_default()
    #     # Predict
    #     predictions = self.predict(pil_image)
        
    #     # Draw boxes on the frame
    #     for element in range(len(predictions[0]['boxes'])):
    #         score = predictions[0]['scores'][element] if 'scores' in predictions[0] else predictions[0]['confidences'][element]
    #         if score > threshold:
    #             box = predictions[0]['boxes'][element].tolist() if 'boxes' in predictions[0] else predictions[0]['xyxy'][element].tolist()
    #             label_idx = predictions[0]['labels'][element].item() if 'labels' in predictions[0] else predictions[0]['classes'][element].item()
    #             label = self.class_names.get(label_idx, 'Unknown')
    #             # Draw rectangle on the frame
    #             cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
    #             text = f"{label}: {score:.2f}"
    #             text_size = cv2.textsize(text, font=font)
    #             text_background = [box[0], box[1], box[0] + text_size[0], box[1] + text_size[1]]
    #             cv2.rectangle(text_background, fill='red')
    #             cv2.text((box[0], box[1]), text, fill='white', font=font)
    #     return frame
    def predict_and_draw_frame(self, frame, threshold=0.2):
    # Convert frame to PIL image for prediction
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Predict using the PIL image
        predictions = self.predict(pil_image)

        # Draw boxes on the frame using OpenCV
        for element in range(len(predictions[0]['boxes'])):
            score = predictions[0]['scores'][element] if 'scores' in predictions[0] else predictions[0]['confidences'][element]
            if score > threshold:
                box = predictions[0]['boxes'][element].tolist() if 'boxes' in predictions[0] else predictions[0]['xyxy'][element].tolist()
                label_idx = predictions[0]['labels'][element].item() if 'labels' in predictions[0] else predictions[0]['classes'][element].item()
                label = self.class_names.get(label_idx, 'Unknown')

                # Draw the bounding box
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

                # Prepare the text for the label and score
                text = f"{label}: {score:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                # Draw the text background
                cv2.rectangle(frame, (int(box[0]), int(box[1] - 20)), (int(box[0]) + text_width, int(box[1])), (0, 255, 0), -1)

                # Put the text on the frame
                cv2.putText(frame, text, (int(box[0]), int(box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


        return frame


# Usage example
# model_path = 'tracker/model_weights.pth'
# detector = HelmetPredictor(model_path)

# image_path = 'archive/train/images/new25.jpg'
# image_with_boxes = detector.predict_and_draw(image_path)

# # Display the image
# image_with_boxes.show()

# # Or save the image
# image_with_boxes.save('image_with_boxes.jpg')
