import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import cv2


class HelmetPredictor:
    def __init__(self, model_path, num_classes=5, model_type='fasterrcnn'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_type == 'fasterrcnn':
            self.model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        elif model_type == 'yolov8':
            # Load YOLOv8 model
            self.model = torch.hub.load('ultralytics/yolov8', 'custom', path=model_path, source='local')  # Ensure path and source are correct
            self.model = YOLO("model/runs/detect/yolov8n_custom3/weights/best.pt")
        else:
            raise ValueError("Unsupported model type: " + model_type)

        self.model.to(self.device).eval()
        self.class_names = {0: 'WithHelmet', 1: 'Without Helmet', 2: 'Unknown', 3: 'NumberPlate', 4: 'Rider'}

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

    def draw_boxes(self, image, predictions, threshold=0.5):
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        
        for element in range(len(predictions[0]['boxes'])):
            score = predictions[0]['scores'][element] if 'scores' in predictions[0] else predictions[0]['confidences'][element]
            if score > threshold:
                box = predictions[0]['boxes'][element].tolist() if 'boxes' in predictions[0] else predictions[0]['xyxy'][element].tolist()
                label_idx = predictions[0]['labels'][element].item() if 'labels' in predictions[0] else predictions[0]['classes'][element].item()
                label = self.class_names.get(label_idx, 'Unknown')
                
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

# image_path = 'archive/val/images/new26.jpg'
# image_with_boxes = detector.predict_and_draw(image_path)

# # Display the image
# image_with_boxes.show()

# # Or save the image
# image_with_boxes.save('image_with_boxes.jpg')
