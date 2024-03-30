from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    with Image.open('archive/train/images/new130.jpg') as img:
        img = img.convert('RGB')
except:
    print("Unidentify")
