import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from helmet_tracker import HelmetPlateDataset





def main():
    dataset = HelmetPlateDataset('../Helmet_Only_dataset/train/images', '../Helmet_Only_dataset/train/labels', transforms=HelmetPlateDataset.get_transform())
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    
    model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=4)  # Assuming 4 classes + background
    model.load_state_dict(torch.load('model_weights.pth'))

    num_classes = 4  # Updating the classifier for new number of classes
    model.trainable = False
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    print(in_features)
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for param in model.parameters():
        param.requires_grad = False


    for param in model.roi_heads.parameters():
        param.requires_grad = True
    print("here")
# Training loop
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)
    for epoch in range(10):
        model.train()
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        print(f'Epoch: {epoch}, Loss: {losses.item()}')
    torch.save(model.state_dict(), 'model_weights_final.pth')

if __name__ == "__main__":
    main()