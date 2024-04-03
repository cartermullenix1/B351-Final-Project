from PIL import Image, ImageDraw, ImageFont

# Define class names based on your classes
class_names = {0: 'WithHelmet', 1: 'Without Helmet', 2: 'Unknown', 3: 'NumberPlate', 4: 'Rider'}

def draw_boxes(image, predictions):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    iw, ih = image.size

    for element in predictions:
        # Extract data from each line
        class_id, cx, cy, bw, bh = element
        label = class_names.get(class_id, 'Unknown')

        # Convert normalized coordinates to pixel coordinates
        x1 = (cx - bw / 2) * iw
        y1 = (cy - bh / 2) * ih
        x2 = (cx + bw / 2) * iw
        y2 = (cy + bh / 2) * ih

        # Draw the bounding box
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)

        # Draw the label
        text = f"{label}"
        text_size = draw.textsize(text, font=font)
        text_background = [x1, y1 - text_size[1], x1 + text_size[0], y1]
        draw.rectangle(text_background, fill='red')
        draw.text((x1, y1 - text_size[1]), text, fill='white', font=font)

    return image

def parse_label_file(label_path, image_size):
    with open(label_path, 'r') as file:
        lines = file.readlines()

    predictions = []
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        cx, cy, bw, bh = [float(num) for num in parts[1:]]
        predictions.append((class_id, cx, cy, bw, bh))

    return predictions

# Load your image and label file
image_path = 'path/to/your/image.jpg'
label_path = 'path/to/your/label.txt'
image = Image.open(image_path)
iw, ih = image.size

# Parse label file
predictions = parse_label_file(label_path, (iw, ih))

# Draw boxes on the image
image_with_boxes = draw_boxes(image, predictions)

# Display the image
image_with_boxes.show()

# Optionally, save the image to a file
image_with_boxes.save('output_image.jpg')
