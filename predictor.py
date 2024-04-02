import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont

class HelmetPredictor:
    def __init__(self, model_path, num_classes=5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        self.class_names = {0: 'Background', 1: 'Without Helmet', 2: 'Helmet', 3: 'NumberPlate', 4: 'Rider'}

    def transform_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = F.to_tensor(image)
        return image.unsqueeze(0)  # Add a batch dimension

    def predict(self, image_path):
        image_tensor = self.transform_image(image_path).to(self.device)
        with torch.no_grad():
            output = self.model(image_tensor)
        return output

    def draw_boxes(self, image, predictions, threshold=0.5):
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        for element in range(len(predictions[0]['boxes'])):
            score = predictions[0]['scores'][element]
            if score > threshold:
                box = predictions[0]['boxes'][element].tolist()
                label_idx = predictions[0]['labels'][element].item()
                label = self.class_names.get(label_idx, 'Unknown')
                draw.rectangle(box, outline='red', width=3)
                text = f"{label}: {score:.2f}"
                text_size = draw.textsize(text, font=font)
                text_background = [box[0], box[1], box[0] + text_size[0], box[1] + text_size[1]]
                draw.rectangle(text_background, fill='red')
                draw.text((box[0], box[1]), text, fill='white', font=font)

        return image

    def predict_and_draw(self, image_path, threshold=0.5):
        predictions = self.predict(image_path)
        image = Image.open(image_path).convert('RGB')
        return self.draw_boxes(image, predictions, threshold)

# Usage example
model_path = 'tracker/model_weights.pth'
detector = HelmetPredictor(model_path)

image_path = 'archive/val/images/new26.jpg'
image_with_boxes = detector.predict_and_draw(image_path)

# Display the image
image_with_boxes.show()

# Or save the image
image_with_boxes.save('image_with_boxes.jpg')
