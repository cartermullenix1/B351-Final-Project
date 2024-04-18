import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image,ImageDraw, ImageFont

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Recreate the model instance
# model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=5)

# # # Load the model weights onto the correct device
# model.load_state_dict(torch.load('tracker/model_weights.pth', map_location=device))

#model = torch.hub.load('.', 'custom', path="model/runs/detect/yolov8n_custom3/weights/best.pt", source='local')
model = torch.load('model/runs/detect/yolov8n_custom3/weights/last.pt')['model']
print("model", model)
# Move model to the right device and set it to evaluation mode
model = model.to(device).float()
model.eval()

# Function to transform input image
def transform_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = F.to_tensor(image)
    return image.unsqueeze(0)  # Add a batch dimension

# Function to perform prediction
def predict(image_path, model, device):
    image_tensor = transform_image(image_path).to(device)
    image_tensor = image_tensor.float()
    with torch.no_grad():
        output = model(image_tensor)
    return output

# Usage
image_path = 'archive/train/images/new3.jpg'
predictions = predict(image_path, model, device)
print(predictions)

#class_names = {0: 'With Helmet', 1: 'Without Helmet', 2: "Rider", 3: 'NumberPlate', 4: 'Background'}
class_names = {0: 'With Helmet', 1: 'Without Helmet', 2: "Rider", 3: 'NumberPlate'}
def draw_boxes(image, predictions, threshold=0.5):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for element in range(len(predictions[0]['boxes'])):
        score = predictions[0]['scores'][element]
        if score > threshold:
            box = predictions[0]['boxes'][element].tolist()
            label_idx = predictions[0]['labels'][element].item()
            print("Label idx", label_idx)  # Get the label index
            label = class_names.get(label_idx, 'Unknown')  # Get the label name
            print("label", label)
            # Draw the bounding box
            draw.rectangle(box, outline='red', width=3)
            
            # Draw the label and score
            text = f"{label}: {score:.2f}"
            text_size = draw.textsize(text, font=font)
            text_background = [box[0], box[1], box[0] + text_size[0], box[1] + text_size[1]]
            draw.rectangle(text_background, fill='red')
            draw.text((box[0], box[1]), text, fill='white', font=font)

    return image

# Function to perform prediction and draw bounding boxes
def predict_and_draw(image_path, model, device):
    image_tensor = transform_image(image_path).to(device)
    
    # Get predictions
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # Draw boxes on the original image
    image = Image.open(image_path).convert('RGB')
    image_with_boxes = draw_boxes(image, predictions)

    return image_with_boxes

# Usage

image_with_boxes = predict_and_draw(image_path, model, device)

# Display the image
image_with_boxes.show()

# Or save the image
image_with_boxes.save('image_with_boxes.jpg')